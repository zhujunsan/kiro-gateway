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
CodeWhisperer profile discovery for AWS SSO OIDC (enterprise) accounts.

Enterprise Kiro IDE credential files frequently omit ``profileArn`` while the
runtime chat API requires it. The OIDC token refresh response does not carry
an ARN either (unlike Kiro Desktop refresh). ``ListAvailableProfiles`` returns
the profiles that belong to the current bearer token, so the gateway can find
the ARN itself once it holds a valid (freshly refreshed) access token.

The SSO region stored in the credential file is where OIDC lives; it is not
necessarily where profiles live. Lookup therefore tries the configured profile
region, the current API region, and then the regions Kiro serves profiles from.
"""

import re
from enum import Enum
from typing import Iterable, List, Optional, Set

import httpx
from loguru import logger

from kiro.config import get_kiro_q_host
from kiro.proxy import resolve_proxy


# Regions where Kiro hosts CodeWhisperer profiles, in lookup priority order.
KIRO_PROFILE_REGIONS: tuple = ("us-east-1", "eu-central-1")

# Per-request timeout for ListAvailableProfiles (seconds).
PROFILE_LOOKUP_TIMEOUT: float = 10.0

# Upper bound on pagination so a broken nextToken cannot hang startup.
PROFILE_LOOKUP_MAX_PAGES: int = 10

# HTTP statuses that mean the bearer token was rejected for this lookup.
_AUTH_REJECTED_STATUSES = frozenset({401, 403})

_LIST_PROFILES_TARGET = "AmazonCodeWhispererService.ListAvailableProfiles"

_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-\d+$")
_PROFILE_ARN_RE = re.compile(
    r"^arn:[a-z0-9-]+:codewhisperer:"
    r"(?P<region>[a-z]{2}(?:-[a-z0-9]+)+-\d+):"
    r"\d{12}:profile/[^/\s]+$"
)


class ProfileDiscoveryFailure(Enum):
    """Reason a profile could not be selected for the current token."""

    UNAVAILABLE = "unavailable"   # network / malformed responses in every region
    AUTH_REJECTED = "auth_rejected"  # every reachable region rejected the token
    # Lookups succeeded but returned no profile. AWS also answers this way
    # (200, empty list) for an invalid token, so it can hide a stale login.
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"       # several profiles and no verified preference


class ProfileDiscoveryError(RuntimeError):
    """
    Raised when no unambiguous profile ARN can be determined.

    Attributes:
        failure: Classified reason, so callers can decide whether a previously
            configured ARN may still be used.
    """

    def __init__(self, message: str, failure: ProfileDiscoveryFailure) -> None:
        """
        Initialize the error.

        Args:
            message: User-facing, actionable explanation (never contains tokens).
            failure: Classified failure reason.
        """
        super().__init__(message)
        self.failure = failure

    @property
    def auth_failure(self) -> bool:
        """Whether the failure means the user must sign in again."""
        return self.failure is ProfileDiscoveryFailure.AUTH_REJECTED


class _RegionLookupError(Exception):
    """Internal: one region's lookup failed."""

    def __init__(self, auth_rejected: bool) -> None:
        super().__init__("region lookup failed")
        self.auth_rejected = auth_rejected


def profile_arn_region(profile_arn: Optional[str]) -> Optional[str]:
    """
    Extract the region from a structurally valid CodeWhisperer profile ARN.

    Args:
        profile_arn: Candidate ARN.

    Returns:
        Region string, or None when the ARN is absent or malformed.

    Examples:
        >>> profile_arn_region("arn:aws:codewhisperer:us-east-1:123456789012:profile/ABC")
        'us-east-1'
        >>> profile_arn_region("not-an-arn") is None
        True
    """
    if not isinstance(profile_arn, str):
        return None
    match = _PROFILE_ARN_RE.fullmatch(profile_arn.strip())
    return match.group("region") if match else None


def _lookup_regions(configured_arn: Optional[str], api_region: Optional[str]) -> List[str]:
    """
    Build the ordered, de-duplicated list of regions to query.

    Args:
        configured_arn: Previously selected ARN (its region is tried first).
        api_region: Region the gateway currently uses for API calls.

    Returns:
        Validated regions in lookup order.
    """
    ordered: List[str] = []
    for region in (profile_arn_region(configured_arn), api_region, *KIRO_PROFILE_REGIONS):
        if region and _REGION_RE.fullmatch(region) and region not in ordered:
            ordered.append(region)
    return ordered


async def _list_profiles_in_region(
    client: httpx.AsyncClient, access_token: str, region: str
) -> Set[str]:
    """
    Return every valid profile ARN visible to the token in one region.

    Args:
        client: HTTP client (no redirects, so the bearer token never leaves AWS).
        access_token: Current access token.
        region: Validated region to query.

    Returns:
        Set of profile ARNs (possibly empty).

    Raises:
        _RegionLookupError: Transport failure, HTTP error, or malformed response.
    """
    url = f"{get_kiro_q_host(region)}/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-amz-json-1.0",
        "X-Amz-Target": _LIST_PROFILES_TARGET,
    }
    arns: Set[str] = set()
    next_token: Optional[str] = None
    seen_tokens: Set[str] = set()
    try:
        for _ in range(PROFILE_LOOKUP_MAX_PAGES):
            response = await client.post(
                url, headers=headers, json={"nextToken": next_token} if next_token else {}
            )
            if response.status_code != 200:
                # Response bodies may echo request data; never log them.
                logger.warning(
                    f"ListAvailableProfiles failed in {region}: HTTP {response.status_code}"
                )
                raise _RegionLookupError(response.status_code in _AUTH_REJECTED_STATUSES)
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
                logger.warning(f"ListAvailableProfiles returned malformed data in {region}")
                raise _RegionLookupError(False)
            for profile in data["profiles"]:
                if isinstance(profile, dict):
                    arn = profile.get("arn")
                    if profile_arn_region(arn):
                        arns.add(arn.strip())
            raw_next = data.get("nextToken")
            if raw_next is not None and not isinstance(raw_next, str):
                raise _RegionLookupError(False)
            next_token = raw_next.strip() if raw_next else None
            if not next_token:
                return arns
            if next_token in seen_tokens:
                logger.warning(f"ListAvailableProfiles repeated a page token in {region}")
                raise _RegionLookupError(False)
            seen_tokens.add(next_token)
        logger.warning(f"ListAvailableProfiles exceeded {PROFILE_LOOKUP_MAX_PAGES} pages in {region}")
        raise _RegionLookupError(False)
    except (httpx.HTTPError, ValueError) as error:
        # Exception text can contain URLs/headers; log only the type.
        logger.warning(f"ListAvailableProfiles unreachable in {region}: {type(error).__name__}")
        raise _RegionLookupError(False) from None


async def discover_profile_arn(
    access_token: str,
    api_region: Optional[str],
    configured_arn: Optional[str] = None,
    regions: Optional[Iterable[str]] = None,
) -> str:
    """
    Select the profile ARN that belongs to the current access token.

    Selection rules:
    1. A configured ARN is kept only when AWS lists it for this token.
    2. Otherwise exactly one discovered profile is selected.
    3. No profiles, several profiles, or failed lookups raise an error.

    Lookup stops at the first region that returns profiles, so a reachable
    primary region avoids slow timeouts in regions the user never uses.

    Args:
        access_token: Valid bearer token (used only in the Authorization header).
        api_region: Current API region of the gateway.
        configured_arn: Previously selected ARN, if any.
        regions: Optional explicit lookup order (mainly for tests).

    Returns:
        Selected profile ARN.

    Raises:
        ProfileDiscoveryError: No unambiguous profile could be selected.
    """
    lookup_order = list(regions) if regions is not None else _lookup_regions(configured_arn, api_region)
    configured = configured_arn.strip() if isinstance(configured_arn, str) else ""
    candidates: Set[str] = set()
    succeeded = 0
    auth_rejected = 0

    async with httpx.AsyncClient(
        timeout=PROFILE_LOOKUP_TIMEOUT, proxy=resolve_proxy(), follow_redirects=False
    ) as client:
        for region in lookup_order:
            try:
                found = await _list_profiles_in_region(client, access_token, region)
            except _RegionLookupError as error:
                auth_rejected += int(error.auth_rejected)
                continue
            succeeded += 1
            logger.debug(f"ListAvailableProfiles found {len(found)} profile(s) in {region}")
            candidates.update(found)
            if candidates:
                break

    if configured and configured in candidates:
        return configured
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise ProfileDiscoveryError(
            "This Kiro account has multiple profiles. Set PROFILE_ARN to the profile "
            "you want to use, then restart the gateway.",
            ProfileDiscoveryFailure.AMBIGUOUS,
        )
    if succeeded:
        raise ProfileDiscoveryError(
            # AWS answers an invalid/expired token with 200 and an empty list,
            # so "not found" may also mean the login is no longer valid.
            "No Kiro profile was found for the current login. Open Kiro and sign "
            "in again; if that does not help, check that your Kiro subscription "
            "is active, then restart the gateway.",
            ProfileDiscoveryFailure.NOT_FOUND,
        )
    if auth_rejected:
        raise ProfileDiscoveryError(
            "Kiro rejected the current login while looking up your profile. Open Kiro "
            "and sign in again, then restart the gateway.",
            ProfileDiscoveryFailure.AUTH_REJECTED,
        )
    raise ProfileDiscoveryError(
        "Could not reach Kiro to look up your profile. Check your network or proxy "
        "settings, then restart the gateway.",
        ProfileDiscoveryFailure.UNAVAILABLE,
    )
