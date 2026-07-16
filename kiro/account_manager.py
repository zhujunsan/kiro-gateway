# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Unified Account System for Kiro Gateway.

Manages multiple Kiro accounts with intelligent failover, sticky behavior,
and circuit breaker pattern for reliability.

Key features:
- Lazy initialization (only first working account at startup)
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- Probabilistic retry for "dead" accounts
- On-demand model discovery only for model-list requests
- Atomic state persistence
"""

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver, normalize_model_name
from kiro.config import (
    HIDDEN_MODELS,
    MODEL_ALIASES,
    HIDDEN_FROM_LIST,
    ACCOUNT_RECOVERY_TIMEOUT,
    ACCOUNT_MAX_BACKOFF_MULTIPLIER,
    ACCOUNT_PROBABILISTIC_RETRY_CHANCE,
    MODEL_DISCOVERY_CACHE_TTL_SECONDS,
    STATE_SAVE_INTERVAL_SECONDS,
    FALLBACK_MODELS,
)
from kiro.account_errors import ErrorType
from kiro.http_client import KiroHttpClient



class ModelDiscoveryError(RuntimeError):
    """Raised when ListAvailableModels returns unusable data."""


def _format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "30s", "5m", "2h", "1d")
    
    Examples:
        >>> _format_duration(30)
        '30s'
        >>> _format_duration(300)
        '5m'
        >>> _format_duration(7200)
        '2h'
        >>> _format_duration(86400)
        '1d'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    else:
        return f"{int(seconds / 86400)}d"


@dataclass
class AccountStats:
    """
    Statistics for account usage.
    
    Tracks request counts for monitoring and future web UI.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


@dataclass
class Account:
    """
    Complete account entity with all dependencies.
    
    Represents a single Kiro account with its authentication,
    model cache, resolver, and runtime state.
    
    Attributes:
        id: Unique identifier (path to credentials file)
        auth_manager: Authentication manager (lazy initialized)
        model_cache: Model metadata cache (lazy initialized)
        model_resolver: Model resolver (lazy initialized)
        failures: Consecutive failure count (for Circuit Breaker)
        last_failure_time: Timestamp of last failure
        models_cached_at: Timestamp when cached model data was last replaced
        model_discovery_attempted_at: Timestamp of the latest list-request discovery attempt
        model_discovery_succeeded: Whether the current cache came from upstream
        stats: Usage statistics
    """
    id: str
    auth_manager: Optional[KiroAuthManager] = None
    model_cache: Optional[ModelInfoCache] = None
    model_resolver: Optional[ModelResolver] = None
    failures: int = 0
    last_failure_time: float = 0.0
    models_cached_at: float = 0.0
    model_discovery_attempted_at: float = 0.0
    model_discovery_succeeded: bool = False
    initialization_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )
    model_discovery_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )
    stats: AccountStats = field(default_factory=AccountStats)


@dataclass
class ModelAccountList:
    """
    List of accounts for a specific model.
    
    Attributes:
        accounts: List of account IDs that have this model
    
    Note: next_index removed - now using global _current_account_index
    """
    accounts: List[str] = field(default_factory=list)


class AccountManager:
    """
    Manages multiple Kiro accounts with intelligent failover.
    
    Responsibilities:
    - Load credentials from credentials.json
    - Lazy initialization of accounts
    - Select next available account (Circuit Breaker + Sticky)
    - Track statistics and failures
    - Persist state to state.json
    
    Example:
        >>> manager = AccountManager("credentials.json", "state.json")
        >>> await manager.load_credentials()
        >>> await manager.load_state()
        >>> account = await manager.get_next_account("claude-opus-4.5")
        >>> await manager.report_success(account.id, "claude-opus-4.5")
    """
    
    def __init__(self, credentials_file: str, state_file: str):
        """
        Initialize AccountManager.
        
        Args:
            credentials_file: Path to credentials.json
            state_file: Path to state.json
        """
        self._credentials_file = credentials_file
        self._state_file = state_file
        self._accounts: Dict[str, Account] = {}
        self._model_to_accounts: Dict[str, ModelAccountList] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._credentials_config: List[Dict] = []
        self._current_account_index: int = 0  # GLOBAL sticky index for all models
    
    async def load_credentials(self) -> None:
        """
        Load credentials from credentials.json.
        
        Validates each entry and creates Account objects.
        Invalid entries are skipped with warnings.
        Folders are scanned for credential files.
        """
        creds_path = Path(self._credentials_file).expanduser()
        
        if not creds_path.exists():
            logger.warning(f"Credentials file not found: {self._credentials_file}")
            return
        
        try:
            with open(creds_path, 'r', encoding='utf-8') as f:
                self._credentials_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return
        
        # Process each credential entry
        for entry in self._credentials_config:
            cred_type = entry.get("type")
            path = entry.get("path")
            enabled = entry.get("enabled", True)
            
            if not enabled:
                continue
            
            # Validate required fields based on type
            if not cred_type:
                logger.warning(f"Invalid credential entry (missing type): {entry}")
                continue
            
            # For json/sqlite types, path is required
            if cred_type in ("json", "sqlite") and not path:
                logger.warning(f"Invalid credential entry (type={cred_type} requires path): {entry}")
                continue
            
            # For refresh_token type, refresh_token field is required
            if cred_type == "refresh_token" and not entry.get("refresh_token"):
                logger.warning(f"Invalid credential entry (type=refresh_token requires refresh_token field): {entry}")
                continue
            
            # Handle refresh_token type (no path processing needed)
            if cred_type == "refresh_token":
                # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                token = entry.get('refresh_token', '')
                token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                account_id = f"refresh_token_{token_hash}"
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
                continue  # Skip path processing for refresh_token
            
            # Handle folder scanning for json/sqlite types
            expanded_path = Path(path).expanduser()
            if expanded_path.is_dir():
                logger.info(f"Scanning folder for credentials: {path}")
                for file_path in expanded_path.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    # Validate file before adding as account
                    account_id = str(file_path.resolve())
                    is_valid = False
                    
                    # Try JSON validation
                    if cred_type == "json":
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Valid if has refreshToken or clientId
                                if 'refreshToken' in data or 'clientId' in data:
                                    is_valid = True
                        except Exception as e:
                            logger.warning(f"Invalid JSON credentials file {file_path.name}: {e}")
                    
                    # Try SQLite validation
                    elif cred_type == "sqlite":
                        try:
                            import sqlite3
                            conn = sqlite3.connect(str(file_path))
                            cursor = conn.cursor()
                            # Check if auth_kv table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_kv'")
                            if cursor.fetchone():
                                is_valid = True
                            conn.close()
                        except Exception as e:
                            logger.warning(f"Invalid SQLite database file {file_path.name}: {e}")
                    
                    if is_valid:
                        self._accounts[account_id] = Account(id=account_id)
                        logger.debug(f"Added account from folder: {account_id}")
                    else:
                        logger.warning(f"Skipping invalid credentials file: {file_path.name}")
            elif expanded_path.is_file() or cred_type == "refresh_token":
                # Single file or refresh_token type
                if cred_type == "refresh_token":
                    # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                    token = entry.get('refresh_token', '')
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    account_id = f"refresh_token_{token_hash}"
                else:
                    account_id = str(expanded_path.resolve())
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
            else:
                logger.warning(f"Credential path not found: {path}")
        
        logger.info(f"Loaded {len(self._accounts)} account(s) from credentials")
    
    async def load_state(self) -> None:
        """
        Load runtime state from state.json.
        
        Restores model_to_accounts mapping and account runtime state.
        Creates empty state if file doesn't exist.
        """
        state_path = Path(self._state_file)
        
        if not state_path.exists():
            logger.debug("State file not found, starting with empty state")
            return
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            # Restore global current_account_index
            self._current_account_index = state_data.get("current_account_index", 0)
            
            # Restore model_to_accounts mapping (without next_index)
            for model, data in state_data.get("model_to_accounts", {}).items():
                self._model_to_accounts[model] = ModelAccountList(
                    accounts=data.get("accounts", [])
                )
            
            # Restore account runtime state
            for account_id, data in state_data.get("accounts", {}).items():
                if account_id in self._accounts:
                    account = self._accounts[account_id]
                    account.failures = data.get("failures", 0)
                    account.last_failure_time = data.get("last_failure_time", 0.0)
                    account.models_cached_at = data.get("models_cached_at", 0.0)
                    
                    stats_data = data.get("stats", {})
                    account.stats = AccountStats(
                        total_requests=stats_data.get("total_requests", 0),
                        successful_requests=stats_data.get("successful_requests", 0),
                        failed_requests=stats_data.get("failed_requests", 0)
                    )
            
            logger.info(f"Loaded state: {len(self._model_to_accounts)} model mappings, {len(self._accounts)} accounts")
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self) -> None:
        """
        Save runtime state to state.json atomically.
        
        Uses tmp file + rename for atomic write.
        """
        state_data = {
            "current_account_index": self._current_account_index,
            "accounts": {
                account_id: {
                    "failures": account.failures,
                    "last_failure_time": account.last_failure_time,
                    "models_cached_at": account.models_cached_at,
                    "stats": {
                        "total_requests": account.stats.total_requests,
                        "successful_requests": account.stats.successful_requests,
                        "failed_requests": account.stats.failed_requests
                    }
                }
                for account_id, account in self._accounts.items()
            },
            "model_to_accounts": {
                model: {
                    "accounts": mal.accounts
                }
                for model, mal in self._model_to_accounts.items()
            }
        }
        
        state_path = Path(self._state_file)
        tmp_path = state_path.with_suffix('.json.tmp')
        
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            tmp_path.replace(state_path)
            logger.debug("State saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    async def save_state_periodically(self) -> None:
        """
        Background task for periodic state saving.
        
        Saves state every STATE_SAVE_INTERVAL_SECONDS if dirty flag is set.
        """
        while True:
            await asyncio.sleep(STATE_SAVE_INTERVAL_SECONDS)
            
            if self._dirty:
                async with self._lock:
                    await self._save_state()
                    self._dirty = False
    
    async def _fetch_available_models(
        self, auth_manager: KiroAuthManager
    ) -> List[Dict[str, Any]]:
        """Fetch and validate the model list from the dedicated Q endpoint.

        Args:
            auth_manager: Account authentication and endpoint provider.

        Returns:
            Valid model metadata returned by ListAvailableModels.

        Raises:
            HTTPException: If the HTTP client exhausts network retries.
            httpx.HTTPError: If authentication or transport handling fails.
            ModelDiscoveryError: If the endpoint returns an error or malformed list.
            ValueError: If the response body is not valid JSON.
        """
        params: Dict[str, str] = {"origin": "AI_EDITOR"}
        if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
            params["profileArn"] = auth_manager.profile_arn

        http_client = KiroHttpClient(auth_manager, shared_client=None)
        try:
            response = await http_client.request_with_retry(
                method="GET",
                url=f"{auth_manager.q_host}/ListAvailableModels",
                json_data=None,
                params=params,
                stream=False,
            )
            if response.status_code != 200:
                raise ModelDiscoveryError(
                    f"ListAvailableModels returned HTTP {response.status_code}"
                )

            data = response.json()
            models = data.get("models") if isinstance(data, dict) else None
            if not isinstance(models, list) or not models:
                raise ModelDiscoveryError(
                    "ListAvailableModels returned an empty or invalid models list"
                )
            if not all(
                isinstance(model, dict)
                and isinstance(model.get("modelId"), str)
                and model["modelId"]
                for model in models
            ):
                raise ModelDiscoveryError(
                    "ListAvailableModels returned malformed model metadata"
                )

            logger.info(
                f"Dynamic model discovery succeeded with {len(models)} models"
            )
            return models
        finally:
            await http_client.close()

    async def _initialize_account(self, account_id: str) -> bool:
        """Initialize authentication and fallback model metadata without discovery.

        Initialization deliberately does not call ``ListAvailableModels``. The
        fallback cache supplies resolver and token-limit metadata so generation
        works before any client asks for a model list. Discovery freshness is
        tracked separately and remains unset until a model-list request.

        Args:
            account_id: Account ID to initialize.

        Returns:
            True if initialization succeeded, otherwise False.
        """
        account = self._accounts.get(account_id)
        if not account:
            return False

        try:
            creds_config: Optional[Dict[str, Any]] = None
            for entry in self._credentials_config:
                path = entry.get("path", "")
                expanded_path = Path(path).expanduser()

                if entry.get("type") == "refresh_token":
                    token = entry.get("refresh_token", "")
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    if account_id == f"refresh_token_{token_hash}":
                        creds_config = entry
                        break
                elif (
                    str(expanded_path.resolve()) == account_id
                    or (
                        expanded_path.is_dir()
                        and account_id.startswith(str(expanded_path.resolve()) + os.sep)
                    )
                ):
                    creds_config = entry
                    break

            if not creds_config:
                logger.error(f"No credentials config found for account: {account_id}")
                return False

            cred_type = creds_config.get("type")
            auth_kwargs: Dict[str, Any] = {
                "profile_arn": creds_config.get("profile_arn"),
                "region": creds_config.get("region", "us-east-1"),
                "api_region": creds_config.get("api_region"),
            }
            if cred_type == "json":
                auth_manager = KiroAuthManager(creds_file=account_id, **auth_kwargs)
            elif cred_type == "sqlite":
                auth_manager = KiroAuthManager(sqlite_db=account_id, **auth_kwargs)
            elif cred_type == "refresh_token":
                auth_manager = KiroAuthManager(
                    refresh_token=creds_config.get("refresh_token"), **auth_kwargs
                )
            else:
                logger.error(f"Unknown credential type: {cred_type}")
                return False

            await auth_manager.get_access_token()

            model_cache = ModelInfoCache()
            await model_cache.update(FALLBACK_MODELS)
            for display_name, internal_id in HIDDEN_MODELS.items():
                model_cache.add_hidden_model(display_name, internal_id)

            model_resolver = ModelResolver(
                cache=model_cache,
                hidden_models=HIDDEN_MODELS,
                aliases=MODEL_ALIASES,
                hidden_from_list=HIDDEN_FROM_LIST,
            )

            account.auth_manager = auth_manager
            account.model_cache = model_cache
            account.model_resolver = model_resolver
            account.models_cached_at = time.time()
            account.model_discovery_attempted_at = 0.0
            account.model_discovery_succeeded = False

            self._update_model_account_mapping(account_id, model_resolver)
            available_models = model_resolver.get_available_models()
            logger.info(
                f"Initialized account with fallback model metadata: "
                f"{account_id} ({len(available_models)} models); discovery deferred"
            )
            self._dirty = True
            return True
        except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
            logger.error(f"Failed to initialize account {account_id}: {error}")
            return False

    async def _ensure_account_initialized(self, account_id: str) -> bool:
        """Single-flight account initialization for model-list requests.

        Args:
            account_id: Account ID to initialize if needed.

        Returns:
            True when the account is initialized and ready.
        """
        account = self._accounts.get(account_id)
        if not account:
            return False
        if account.auth_manager is not None:
            return True

        async with account.initialization_lock:
            if account.auth_manager is not None:
                return True
            return await self._initialize_account(account_id)

    def _update_model_account_mapping(
        self, account_id: str, model_resolver: ModelResolver
    ) -> None:
        """Add resolver-visible models to the account routing index.

        Args:
            account_id: Account that exposes the models.
            model_resolver: Resolver whose output includes aliases and hiding rules.
        """
        for model in model_resolver.get_available_models():
            if model not in self._model_to_accounts:
                self._model_to_accounts[model] = ModelAccountList()
            if account_id not in self._model_to_accounts[model].accounts:
                self._model_to_accounts[model].accounts.append(account_id)

    async def _discover_account_models(self, account_id: str) -> bool:
        """Refresh one account only when a model-list request permits it.

        The per-account lock provides single-flight behavior. Every attempt,
        including a failure, starts a fresh four-hour throttle window. A first
        failure retains the initialization fallback; a later failure retains
        the last successful dynamic list.

        Args:
            account_id: Account ID whose list cache should be checked.

        Returns:
            True when dynamic data replaced the cache, otherwise False.
        """
        account = self._accounts.get(account_id)
        if (
            not account
            or not account.auth_manager
            or not account.model_cache
            or not account.model_resolver
        ):
            return False

        async with account.model_discovery_lock:
            now = time.time()
            age = now - account.model_discovery_attempted_at
            if (
                account.model_discovery_attempted_at > 0
                and age < MODEL_DISCOVERY_CACHE_TTL_SECONDS
            ):
                remaining = MODEL_DISCOVERY_CACHE_TTL_SECONDS - age
                logger.debug(
                    "Model discovery cache is valid; serving cached list without "
                    f"upstream request for another {_format_duration(remaining)}"
                )
                return False

            account.model_discovery_attempted_at = now
            try:
                models_list = await self._fetch_available_models(account.auth_manager)
            except (
                HTTPException,
                httpx.HTTPError,
                ModelDiscoveryError,
                ValueError,
            ) as error:
                account.model_discovery_attempted_at = time.time()
                self._dirty = True
                if account.model_discovery_succeeded:
                    logger.warning(
                        "Dynamic model refresh failed; serving stale dynamic cache "
                        f"for the next four-hour window: {error}"
                    )
                else:
                    logger.warning(
                        "Initial model discovery failed; serving fallback cache "
                        f"for the next four-hour window: {error}"
                    )
                return False

            await account.model_cache.update(models_list)
            for display_name, internal_id in HIDDEN_MODELS.items():
                account.model_cache.add_hidden_model(display_name, internal_id)

            completed_at = time.time()
            account.models_cached_at = completed_at
            account.model_discovery_attempted_at = completed_at
            account.model_discovery_succeeded = True
            self._update_model_account_mapping(account_id, account.model_resolver)
            logger.info(
                f"Dynamic model discovery cached {len(models_list)} models "
                "for four hours"
            )
            self._dirty = True
            return True

    async def get_next_account(self, model: str, exclude_accounts: Optional[set] = None) -> Optional[Account]:
        """
        Get next available account for model (Circuit Breaker + Sticky).
        
        Implements:
        - Sticky behavior (prefer successful account)
        - Circuit Breaker with exponential backoff
        - Probabilistic retry for "dead" accounts (10%)
        - Model discovery only for explicit model-list requests
        - Exclusion of already-tried accounts in current failover loop
        
        Args:
            model: Model name (will be normalized)
            exclude_accounts: Set of account IDs to exclude (already tried in current failover loop)
        
        Returns:
            Account object or None if no accounts available
        """
        async with self._lock:
            # Special case: single account - bypass Circuit Breaker
            # Circuit Breaker is meaningless for single account - user should see real Kiro API errors
            # instead of generic "Account unavailable" after cooldown kicks in
            if len(self._accounts) == 1:
                account_id = list(self._accounts.keys())[0]
                account = self._accounts[account_id]
                
                # Skip if already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    return None
                
                # Lazy initialization if needed
                if account.auth_manager is None:
                    success = await self._ensure_account_initialized(account_id)
                    if not success:
                        return None
                
                # # Validate model availability
                # if account.model_resolver:
                #     normalized_model = normalize_model_name(model)
                #     available_models = account.model_resolver.get_available_models()
                #     if normalized_model not in available_models:
                #         return None
                
                # Always return single account (ignore cooldown/failures)
                # No model validation - let Kiro API decide (gateway, not gatekeeper)
                return account
            
            # Multi-account logic: GLOBAL sticky
            normalized_model = normalize_model_name(model)
            
            # ALWAYS start from GLOBAL index (one current account for ALL models)
            start_index = self._current_account_index
            
            # ALWAYS iterate over ALL accounts
            all_account_ids = list(self._accounts.keys())
            
            for i in range(len(all_account_ids)):
                current_index = (start_index + i) % len(all_account_ids)
                account_id = all_account_ids[current_index]
                account = self._accounts[account_id]
                
                # Skip accounts already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    continue
                
                # Check Circuit Breaker (Half-Open state with exponential backoff)
                if account.failures > 0:
                    time_since_failure = time.time() - account.last_failure_time
                    
                    # Exponential backoff: base * 2^(failures - 1), capped at MAX_MULTIPLIER
                    # 1 failure: 60s, 2: 120s, 3: 240s, ..., 12+: 86400s (1 day cap)
                    backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                    effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                    
                    if time_since_failure < effective_timeout:
                        # Probabilistic retry (10% chance)
                        if random.random() > ACCOUNT_PROBABILISTIC_RETRY_CHANCE:
                            continue
                        else:
                            logger.info(f"Probabilistic retry for broken account {account_id}")
                    else:
                        # Half-Open: recovery timeout passed
                        logger.info(f"Half-Open state for {account_id} (recovery timeout passed, effective={effective_timeout}s)")
                
                # Lazy initialization
                if account.auth_manager is None:
                    success = await self._ensure_account_initialized(account_id)
                    if not success:
                        account.failures += 1
                        self._dirty = True
                        continue
                
                # # Check if model is available on this account
                # available_models = account.model_resolver.get_available_models()
                # if normalized_model not in available_models:
                #     continue
                
                # No model validation - let Kiro API decide (gateway, not gatekeeper)
                # Account is suitable!
                return account
            
            # All accounts unavailable
            return None
    
    async def report_success(self, account_id: str, model: str) -> None:
        """
        Report successful request (reset failures, update stats, sticky, dynamic learning).
        
        Args:
            account_id: Account ID
            model: Model name
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Reset failures
            if account.failures > 0:
                account.failures = 0
                self._dirty = True
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.successful_requests += 1
            self._dirty = True
            
            # Dynamic learning: add model to mapping if successful
            # This allows system to learn about new models not in FALLBACK_MODELS
            normalized_model = normalize_model_name(model)
            if normalized_model not in self._model_to_accounts:
                self._model_to_accounts[normalized_model] = ModelAccountList()
                logger.debug(f"Dynamic learning: discovered new model '{normalized_model}'")
            if account_id not in self._model_to_accounts[normalized_model].accounts:
                self._model_to_accounts[normalized_model].accounts.append(account_id)
                logger.debug(f"Dynamic learning: model '{normalized_model}' works on account {account_id}")
                self._dirty = True
            
            # GLOBAL STICKY: Update global current_account_index
            all_account_ids = list(self._accounts.keys())
            try:
                successful_index = all_account_ids.index(account_id)
                if self._current_account_index != successful_index:
                    self._current_account_index = successful_index
                    self._dirty = True
            except ValueError:
                pass
    
    async def report_failure(
        self,
        account_id: str,
        model: str,
        error_type: ErrorType,
        status_code: int,
        reason: Optional[str]
    ) -> None:
        """
        Report failed request (update failures, stats, failover).
        
        Args:
            account_id: Account ID
            model: Model name
            error_type: Error classification (FATAL or RECOVERABLE)
            status_code: HTTP status code
            reason: Error reason from Kiro API
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Special case: INVALID_MODEL_ID is discovery process, not account failure
            # Account is healthy, model is just not available on this account
            # Log for user visibility but don't penalize account statistics
            if reason == "INVALID_MODEL_ID":
                account.stats.total_requests += 1
                self._dirty = True
                logger.warning(
                    f"Model '{model}' not available on account {account_id}: "
                    f"status={status_code}, reason={reason}"
                )
                return
            
            # Update failure count (only for RECOVERABLE)
            if error_type == ErrorType.RECOVERABLE:
                account.failures += 1
                account.last_failure_time = time.time()
                self._dirty = True
                
                # Calculate backoff for logging
                backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                logger.warning(
                    f"Account {account_id} failure #{account.failures}: "
                    f"status={status_code}, reason={reason}, "
                    f"cooldown={_format_duration(effective_timeout)}"
                )
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.failed_requests += 1
            self._dirty = True
            
            # GLOBAL STICKY: Do NOT change _current_account_index on failure
            # It only changes on success (GLOBAL sticky behavior)
            # Failover happens through exclude_accounts in get_next_account()
    
    def get_first_account(self) -> Account:
        """
        Get first initialized account (for legacy mode).
        
        Returns:
            First initialized account
        
        Raises:
            RuntimeError: If no initialized accounts available
        """
        for account in self._accounts.values():
            if account.auth_manager is not None:
                return account
        raise RuntimeError("No initialized accounts available")
    
    async def get_all_available_models(self) -> List[str]:
        """Initialize accounts and discover models for an explicit list request.

        This is the only public account-manager entry point that may invoke
        ``ListAvailableModels``. Each account is independently single-flight and
        throttled, so network I/O does not hold the manager-wide routing lock.

        Returns:
            Sorted unique model IDs after aliases and hiding rules are applied.
        """
        all_models: set[str] = set()
        for account_id in list(self._accounts):
            if not await self._ensure_account_initialized(account_id):
                continue
            await self._discover_account_models(account_id)
            account = self._accounts[account_id]
            if account.model_resolver:
                all_models.update(account.model_resolver.get_available_models())
        return sorted(all_models)
