# -*- coding: utf-8 -*-

"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- On-demand model discovery cache
- State persistence
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx

from kiro.account_manager import (
    Account,
    AccountStats,
    ModelAccountList,
    AccountManager,
    ModelDiscoveryError,
    _format_duration
)
from kiro.account_errors import ErrorType
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """
    
    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.
        
        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")
        
        # Act
        account = Account(id="/test/path.json")
        
        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")
        
        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)
    
    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.
        
        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")
        
        # Act
        stats = AccountStats()
        
        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.
        
        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.
        
        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.
        
        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")
    
    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.
        
        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.
        
        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")
        
        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1  # Only valid file loaded
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.
        
        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.
        
        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.
        
        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.
        
        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.
        
        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.
        
        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))
        
        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) > 0
    
    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test restoration of global current_account_index.
        
        What it does: Restores sticky index from state
        Purpose: Verify global sticky behavior persistence
        """
        print("\n=== Test: load_state restores current_account_index ===")
        
        # Arrange
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Current account index: {manager._current_account_index}")
        
        assert manager._current_account_index == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.
        
        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")
        
        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")
        
        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).
        
        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")
        
        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())
        
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")
        
        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
    
    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).
        
        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) == 0
        assert manager._current_account_index == 0
    
    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.
        
        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        
        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_initialize_account_uses_fallback_without_discovery(self, tmp_path):
        """Initialization prepares runtime metadata but never calls model discovery."""
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "region": "us-east-1",
        }))
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        manager = AccountManager(str(creds_file), str(tmp_path / "state.json"))
        await manager.load_credentials()
        account_id = str(test_json.resolve())

        with patch("kiro.account_manager.KiroHttpClient") as http_client:
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert success is True
        http_client.assert_not_called()
        assert account.model_cache is not None
        assert account.model_resolver is not None
        assert account.model_discovery_attempted_at == 0.0
        assert account.model_discovery_succeeded is False
        assert "gpt-5.6-sol" in account.model_cache.get_all_model_ids()
        assert "kiro-s-5" in account.model_resolver.get_available_models()
        assert "kiro-o-5" in account.model_resolver.get_available_models()
        assert "auto-kiro" not in account.model_resolver.get_available_models()
        # A successful init must leave no stale failure reason behind.
        assert account.last_init_error is None
        assert account.last_init_auth_failure is False

    @staticmethod
    def _single_json_account(tmp_path) -> tuple[AccountManager, str]:
        """Build a manager with one JSON-credential account, not yet initialized."""
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "expired_access",
            "expiresAt": "2000-01-01T00:00:00.000Z",
            "region": "us-east-1",
        }))
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        manager = AccountManager(str(creds_file), str(tmp_path / "state.json"))
        return manager, str(test_json.resolve())

    @pytest.mark.asyncio
    async def test_initialize_records_auth_failure_cause(self, tmp_path):
        """An expired login must be recorded as an auth failure, with its cause.

        Without this the startup path could only say "failed", which is why
        Sentry KIRO-GATEWAY-TRAY-W was unactionable.
        """
        manager, account_id = self._single_json_account(tmp_path)
        await manager.load_credentials()

        request = httpx.Request("POST", "https://oidc.test/token")
        response = httpx.Response(400, request=request, text="invalid_grant")
        with patch.object(
            KiroAuthManager,
            "get_access_token",
            new=AsyncMock(side_effect=httpx.HTTPStatusError(
                "400", request=request, response=response
            )),
        ):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert success is False
        assert account.last_init_auth_failure is True
        assert "HTTPStatusError" in account.last_init_error

    @pytest.mark.asyncio
    async def test_initialize_records_transient_failure_as_non_auth(self, tmp_path):
        """A proxy failure must not be blamed on the user's credentials."""
        manager, account_id = self._single_json_account(tmp_path)
        await manager.load_credentials()

        with patch.object(
            KiroAuthManager,
            "get_access_token",
            new=AsyncMock(side_effect=httpx.ConnectError("All connection attempts failed")),
        ):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert success is False
        assert account.last_init_auth_failure is False
        assert "ConnectError" in account.last_init_error

    @pytest.mark.asyncio
    async def test_initialize_treats_unrecoverable_refresh_as_auth_failure(self, tmp_path):
        """ValueError from get_access_token means refresh is exhausted."""
        manager, account_id = self._single_json_account(tmp_path)
        await manager.load_credentials()

        with patch.object(
            KiroAuthManager,
            "get_access_token",
            new=AsyncMock(side_effect=ValueError("Token expired and refresh failed")),
        ):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert success is False
        assert account.last_init_auth_failure is True



class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.
        
        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()
        
        # Act
        account = await manager.get_next_account("claude-opus-4.5")
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.
        
        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures
        manager._accounts[account_id].failures = 5
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0
    
    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.
        
        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.
        
        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1
    
    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.
        
        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """
    
    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.
        
        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager._save_state()
        
        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()
        
        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.
        
        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        account = manager.get_first_account()
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None
    
    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.
        
        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerDescribeInitFailure:
    """describe_init_failure() must separate the three failure causes.

    Startup previously raised a single opaque "Failed to initialize any account",
    so neither the user nor Sentry could tell an expired login from an empty
    credentials file from a proxy failure (Sentry KIRO-GATEWAY-TRAY-W).
    """

    @staticmethod
    def _manager(tmp_path) -> AccountManager:
        return AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json"),
        )

    def test_no_accounts_reports_not_configured(self, tmp_path):
        manager = self._manager(tmp_path)

        result = manager.describe_init_failure()

        assert result["code"] == "account_not_configured"
        assert result["account_count"] == 0
        assert result["errors"] == {}
        assert "Sign in to Kiro" in result["message"]

    def test_all_auth_failures_report_auth_required(self, tmp_path):
        manager = self._manager(tmp_path)
        for account_id in ("a", "b"):
            account = Account(id=account_id)
            account.last_init_error = "HTTPStatusError: 400 invalid_grant"
            account.last_init_auth_failure = True
            manager._accounts[account_id] = account

        result = manager.describe_init_failure()

        assert result["code"] == "account_auth_required"
        assert result["account_count"] == 2
        assert set(result["errors"]) == {"a", "b"}
        assert "sign in again" in result["message"]

    def test_transient_failure_is_not_auth_required(self, tmp_path):
        manager = self._manager(tmp_path)
        account = Account(id="a")
        account.last_init_error = "ConnectError: proxy down"
        account.last_init_auth_failure = False
        manager._accounts["a"] = account

        result = manager.describe_init_failure()

        assert result["code"] == "account_init_failed"
        assert "network/proxy" in result["message"]

    def test_mixed_causes_do_not_blame_the_user(self, tmp_path):
        """One transient failure means we must not claim credentials are bad."""
        manager = self._manager(tmp_path)
        auth_account = Account(id="auth")
        auth_account.last_init_error = "HTTPStatusError: 400"
        auth_account.last_init_auth_failure = True
        net_account = Account(id="net")
        net_account.last_init_error = "ConnectError: dns"
        net_account.last_init_auth_failure = False
        manager._accounts = {"auth": auth_account, "net": net_account}

        result = manager.describe_init_failure()

        assert result["code"] == "account_init_failed"
        assert set(result["errors"]) == {"auth", "net"}

    def test_accounts_never_attempted_are_not_reported_as_auth_failures(self, tmp_path):
        """No attempt yet means no error text — must not imply a bad login."""
        manager = self._manager(tmp_path)
        manager._accounts["fresh"] = Account(id="fresh")

        result = manager.describe_init_failure()

        assert result["code"] == "account_init_failed"
        assert result["errors"] == {}


class TestAccountManagerHasInitializedAccount:
    """has_initialized_account() backs the /health readiness flag."""

    @staticmethod
    def _manager(tmp_path) -> AccountManager:
        return AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json"),
        )

    def test_false_with_no_accounts(self, tmp_path):
        assert self._manager(tmp_path).has_initialized_account() is False

    def test_false_when_accounts_lack_auth_manager(self, tmp_path):
        manager = self._manager(tmp_path)
        manager._accounts["a"] = Account(id="a")
        assert manager.has_initialized_account() is False

    def test_true_when_any_account_is_initialized(self, tmp_path):
        manager = self._manager(tmp_path)
        manager._accounts["a"] = Account(id="a")
        ready = Account(id="b")
        ready.auth_manager = Mock()
        manager._accounts["b"] = ready
        assert manager.has_initialized_account() is True


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.
        
        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        models = await manager.get_all_available_models()
        
        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """
    
    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"
    
    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"
    
    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"
    
    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"


class TestAccountManagerOnDemandModelDiscovery:
    """Tests for list-request-only model discovery and four-hour throttling."""

    @staticmethod
    async def _manager(tmp_path):
        test_json = tmp_path / "discovery.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123:profile/test",
            "region": "us-east-1",
        }))
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        manager = AccountManager(str(creds_file), str(tmp_path / "state.json"))
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        assert await manager._initialize_account(account_id)
        return manager, account_id

    @staticmethod
    def _response(model_ids):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "models": [{"modelId": model_id} for model_id in model_ids]
        }
        return response

    @pytest.mark.asyncio
    async def test_chat_selection_never_triggers_discovery(self, tmp_path):
        manager, _ = await self._manager(tmp_path)
        with patch.object(manager, "_fetch_available_models", new=AsyncMock()) as fetch:
            account = await manager.get_next_account("unknown-future-model")
        assert account is not None
        fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_list_request_discovers_then_cache_skips_four_hours(self, tmp_path):
        manager, account_id = await self._manager(tmp_path)
        dynamic_ids = ["auto", "gpt-5.6-sol", "claude-sonnet-5", "claude-sonnet-4.5"]
        with patch.object(
            manager, "_fetch_available_models",
            new=AsyncMock(return_value=[{"modelId": item} for item in dynamic_ids]),
        ) as fetch:
            first = await manager.get_all_available_models()
            second = await manager.get_all_available_models()

        assert fetch.await_count == 1
        assert first == second
        assert "auto" in first
        assert "gpt-5.6-sol" in first
        assert "kiro-s-5" in first
        assert "auto-kiro" not in first
        assert "kiro-5.6-sol" not in first
        assert "claude-sonnet-4.5" not in first
        cache_ids = manager._accounts[account_id].model_cache.get_all_model_ids()
        assert set(cache_ids) == set(dynamic_ids)
        assert "kiro-s-5" not in cache_ids

    @pytest.mark.asyncio
    async def test_expired_dynamic_cache_refreshes_once_and_replaces(self, tmp_path):
        manager, account_id = await self._manager(tmp_path)
        fetch = AsyncMock(side_effect=[
            [{"modelId": "claude-sonnet-5"}],
            [{"modelId": "claude-opus-5"}],
        ])
        with patch.object(manager, "_fetch_available_models", new=fetch):
            first = await manager.get_all_available_models()
            manager._accounts[account_id].model_discovery_attempted_at -= 4 * 60 * 60 + 1
            refreshed = await manager.get_all_available_models()
            repeated = await manager.get_all_available_models()

        assert fetch.await_count == 2
        assert "kiro-s-5" in first
        assert "kiro-o-5" in refreshed
        assert "claude-sonnet-5" not in refreshed
        assert refreshed == repeated

    @pytest.mark.asyncio
    async def test_initial_failure_caches_fallback_with_aliases(self, tmp_path):
        manager, account_id = await self._manager(tmp_path)
        fetch = AsyncMock(side_effect=ModelDiscoveryError("offline"))
        with patch.object(manager, "_fetch_available_models", new=fetch):
            first = await manager.get_all_available_models()
            second = await manager.get_all_available_models()

        assert fetch.await_count == 1
        assert first == second
        assert "gpt-5.6-sol" in first
        assert "auto" in first
        assert "kiro-s-5" in first
        assert "kiro-o-5" in first
        assert "auto-kiro" not in first
        assert manager._accounts[account_id].model_discovery_succeeded is False

    @pytest.mark.asyncio
    async def test_failed_expired_refresh_keeps_stale_and_throttles(self, tmp_path):
        manager, account_id = await self._manager(tmp_path)
        fetch = AsyncMock(side_effect=[
            [{"modelId": "claude-sonnet-5"}, {"modelId": "gpt-5.6-sol"}],
            ModelDiscoveryError("offline"),
        ])
        with patch.object(manager, "_fetch_available_models", new=fetch):
            dynamic = await manager.get_all_available_models()
            manager._accounts[account_id].model_discovery_attempted_at -= 4 * 60 * 60 + 1
            stale = await manager.get_all_available_models()
            repeated = await manager.get_all_available_models()

        assert fetch.await_count == 2
        assert dynamic == stale == repeated
        assert "kiro-s-5" in stale
        assert manager._accounts[account_id].model_discovery_succeeded is True

    @pytest.mark.asyncio
    async def test_concurrent_list_requests_are_single_flight(self, tmp_path):
        manager, _ = await self._manager(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def fetch(_auth_manager):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return [{"modelId": "claude-sonnet-5"}]

        with patch.object(manager, "_fetch_available_models", side_effect=fetch):
            first_task = asyncio.create_task(manager.get_all_available_models())
            await started.wait()
            second_task = asyncio.create_task(manager.get_all_available_models())
            release.set()
            first, second = await asyncio.gather(first_task, second_task)

        assert calls == 1
        assert first == second

    @pytest.mark.asyncio
    async def test_fetch_uses_q_host_profile_rule_and_not_runtime(self, tmp_path):
        manager, account_id = await self._manager(tmp_path)
        account = manager._accounts[account_id]
        response = self._response(["claude-sonnet-5"])
        mock_client = AsyncMock()
        mock_client.request_with_retry = AsyncMock(return_value=response)
        mock_client.close = AsyncMock()

        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            models = await manager._fetch_available_models(account.auth_manager)

        kwargs = mock_client.request_with_retry.await_args.kwargs
        assert kwargs["url"] == "https://q.us-east-1.amazonaws.com/ListAvailableModels"
        assert kwargs["params"]["origin"] == "AI_EDITOR"
        assert "profileArn" in kwargs["params"]
        assert account.auth_manager.api_host == "https://runtime.us-east-1.kiro.dev"
        assert models == [{"modelId": "claude-sonnet-5"}]
