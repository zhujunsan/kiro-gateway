# -*- coding: utf-8 -*-

"""
Unit tests for kiro.profile_discovery.

Covers profile selection rules, multi-region fallback (SSO region differs from
profile region), pagination limits, failure classification, and token hygiene.
All HTTP goes through httpx.MockTransport; no real network access.
"""

import json
from typing import Callable, Dict, List

import httpx
import pytest

from kiro import profile_discovery
from kiro.profile_discovery import (
    KIRO_PROFILE_REGIONS,
    PROFILE_LOOKUP_MAX_PAGES,
    ProfileDiscoveryError,
    ProfileDiscoveryFailure,
    discover_profile_arn,
    profile_arn_region,
)

US_ARN = "arn:aws:codewhisperer:us-east-1:119004746410:profile/USPROFILE"
EU_ARN = "arn:aws:codewhisperer:eu-central-1:222222222222:profile/EUPROFILE"
OLD_ARN = "arn:aws:codewhisperer:us-east-1:111111111111:profile/OLD"

Handler = Callable[[httpx.Request], httpx.Response]

# Captured at import (collection) time, before the global network-blocking
# fixture replaces httpx.AsyncClient with a mock.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture
def install_transport(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], List[httpx.Request]]:
    """
    Replace the module's AsyncClient with a MockTransport-backed client.

    Returns:
        Installer taking a request handler and returning the captured requests.
    """
    real_client = _REAL_ASYNC_CLIENT

    def install(handler: Handler) -> List[httpx.Request]:
        captured: List[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return handler(request)

        def factory(**kwargs: object) -> httpx.AsyncClient:
            kwargs.pop("proxy", None)
            return real_client(**kwargs, transport=httpx.MockTransport(recording))

        monkeypatch.setattr(profile_discovery.httpx, "AsyncClient", factory)
        return captured

    return install


def _by_region(responses: Dict[str, object]) -> Handler:
    """
    Build a handler returning per-region responses keyed by host region.

    A value may be an httpx.Response, a dict (200 JSON), or an exception instance.
    Regions without an entry answer 200 with no profiles.
    """
    def handle(request: httpx.Request) -> httpx.Response:
        region = request.url.host.split(".")[1]
        value = responses.get(region, {"profiles": []})
        if isinstance(value, Exception):
            raise value
        if isinstance(value, httpx.Response):
            return value
        return httpx.Response(200, json=value)

    return handle


class TestProfileArnRegion:
    """Tests for ARN parsing."""

    @pytest.mark.parametrize(
        "arn,region",
        [
            (US_ARN, "us-east-1"),
            (EU_ARN, "eu-central-1"),
            ("  " + US_ARN + "  ", "us-east-1"),
            ("arn:aws-cn:codewhisperer:cn-north-1:123456789012:profile/x", "cn-north-1"),
        ],
    )
    def test_valid_arn_returns_region(self, arn: str, region: str) -> None:
        """
        What it does: Extracts region from structurally valid ARNs.
        Purpose: Region drives where the gateway sends API calls.
        """
        assert profile_arn_region(arn) == region

    @pytest.mark.parametrize(
        "arn",
        [None, 42, [], "", "not-an-arn",
         "arn:aws:iam::123456789012:role/x",
         "arn:aws:codewhisperer:us-east-1:123:profile/x",
         "arn:aws:codewhisperer:us-east-1:123456789012:profile/",
         "arn:aws:codewhisperer:us-east-1:123456789012:profile/a/b",
         "arn:aws:codewhisperer:evil.host:123456789012:profile/x"],
    )
    def test_invalid_arn_returns_none(self, arn: object) -> None:
        """
        What it does: Rejects malformed or wrongly typed ARNs.
        Purpose: An injected region must never become a request host.
        """
        assert profile_arn_region(arn) is None


class TestDiscoverProfileArnSuccess:
    """Selection rules for successful lookups."""

    @pytest.mark.asyncio
    async def test_single_profile_selected_and_token_sent_only_in_header(self, install_transport) -> None:
        """
        What it does: Selects the only profile and sends the correct AWS JSON request.
        Purpose: Fresh enterprise users need no manual ARN.
        """
        requests = install_transport(_by_region({"us-east-1": {"profiles": [{"arn": US_ARN}]}}))

        arn = await discover_profile_arn("secret-token", "us-east-1")

        assert arn == US_ARN
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == "https://q.us-east-1.amazonaws.com/"
        assert request.headers["Authorization"] == "Bearer secret-token"
        assert request.headers["X-Amz-Target"] == "AmazonCodeWhispererService.ListAvailableProfiles"
        assert json.loads(request.content) == {}
        assert "secret-token" not in str(request.url)

    @pytest.mark.asyncio
    async def test_sso_region_without_profiles_falls_back_to_kiro_regions(self, install_transport) -> None:
        """
        What it does: SSO region (ap-northeast-1) has no profiles; us-east-1 does.
        Purpose: Regression for using the OIDC region as the profile region.
        """
        requests = install_transport(_by_region({
            "ap-northeast-1": httpx.ConnectError("tls reset"),
            "us-east-1": {"profiles": [{"arn": US_ARN}]},
        }))

        arn = await discover_profile_arn("tok", "ap-northeast-1")

        assert arn == US_ARN
        hosts = [r.url.host for r in requests]
        assert hosts == ["q.ap-northeast-1.amazonaws.com", "q.us-east-1.amazonaws.com"]

    @pytest.mark.asyncio
    async def test_profile_in_secondary_kiro_region_is_found(self, install_transport) -> None:
        """
        What it does: Profile lives only in eu-central-1.
        Purpose: Non-US Kiro tenants must also work without manual input.
        """
        install_transport(_by_region({"eu-central-1": {"profiles": [{"arn": EU_ARN}]}}))

        assert await discover_profile_arn("tok", "us-east-1") == EU_ARN

    @pytest.mark.asyncio
    async def test_lookup_stops_at_first_region_with_profiles(self, install_transport) -> None:
        """
        What it does: Stops after the first region that returns profiles.
        Purpose: Avoid slow timeouts in regions the user never uses.
        """
        requests = install_transport(_by_region({
            "us-east-1": {"profiles": [{"arn": US_ARN}]},
            "eu-central-1": {"profiles": [{"arn": EU_ARN}]},
        }))

        assert await discover_profile_arn("tok", "us-east-1") == US_ARN
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_configured_arn_region_is_queried_first_and_kept_when_listed(self, install_transport) -> None:
        """
        What it does: A configured EU ARN is verified in its own region first.
        Purpose: Respect a verified manual choice among multiple profiles.
        """
        other = EU_ARN.replace("EUPROFILE", "OTHER")
        requests = install_transport(_by_region({
            "eu-central-1": {"profiles": [{"arn": other}, {"arn": EU_ARN}]},
        }))

        assert await discover_profile_arn("tok", "us-east-1", EU_ARN) == EU_ARN
        assert requests[0].url.host == "q.eu-central-1.amazonaws.com"

    @pytest.mark.asyncio
    async def test_stale_configured_arn_is_replaced_by_current_profile(self, install_transport) -> None:
        """
        What it does: The configured ARN is not listed for this token.
        Purpose: Account switches must not keep the previous account's ARN.
        """
        install_transport(_by_region({"us-east-1": {"profiles": [{"arn": US_ARN}]}}))

        assert await discover_profile_arn("tok", "us-east-1", OLD_ARN) == US_ARN

    @pytest.mark.asyncio
    async def test_malformed_entries_are_ignored(self, install_transport) -> None:
        """
        What it does: Skips non-dict entries and invalid ARNs.
        Purpose: Upstream schema drift must not crash selection.
        """
        install_transport(_by_region({"us-east-1": {"profiles": [
            None, 1, {"arn": []}, {"arn": "bad"}, {}, {"arn": US_ARN}, {"arn": US_ARN},
        ]}}))

        assert await discover_profile_arn("tok", "us-east-1") == US_ARN

    @pytest.mark.asyncio
    async def test_pagination_is_followed(self, install_transport) -> None:
        """
        What it does: Collects profiles across pages before selecting.
        Purpose: A later page may confirm the configured ARN.
        """
        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if not body:
                return httpx.Response(200, json={"profiles": [{"arn": OLD_ARN}], "nextToken": "p2"})
            assert body == {"nextToken": "p2"}
            return httpx.Response(200, json={"profiles": [{"arn": US_ARN}], "nextToken": None})

        requests = install_transport(handle)

        assert await discover_profile_arn("tok", "us-east-1", US_ARN) == US_ARN
        assert len(requests) == 2


class TestDiscoverProfileArnErrors:
    """Failure classification; each class drives a different user action."""

    @pytest.mark.asyncio
    async def test_multiple_profiles_without_preference_is_ambiguous(self, install_transport) -> None:
        """
        What it does: Two profiles, no configured choice.
        Purpose: Never guess between profiles.
        """
        install_transport(_by_region({"us-east-1": {"profiles": [{"arn": US_ARN}, {"arn": OLD_ARN}]}}))

        with pytest.raises(ProfileDiscoveryError) as caught:
            await discover_profile_arn("tok", "us-east-1")
        assert caught.value.failure is ProfileDiscoveryFailure.AMBIGUOUS
        assert "PROFILE_ARN" in str(caught.value)

    @pytest.mark.asyncio
    async def test_no_profiles_anywhere_is_not_found(self, install_transport) -> None:
        """
        What it does: Every region answers with an empty list.
        Purpose: Distinguish "no subscription" from network failures.
        """
        requests = install_transport(_by_region({}))

        with pytest.raises(ProfileDiscoveryError) as caught:
            await discover_profile_arn("tok", "us-east-1")
        assert caught.value.failure is ProfileDiscoveryFailure.NOT_FOUND
        assert len(requests) == len(KIRO_PROFILE_REGIONS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_token_rejected_everywhere_is_auth_failure(self, install_transport, status: int) -> None:
        """
        What it does: All regions reject the bearer token.
        Purpose: User must sign in again; surfaced as an auth failure.
        """
        install_transport(lambda r: httpx.Response(status, text="private-body"))

        with pytest.raises(ProfileDiscoveryError) as caught:
            await discover_profile_arn("secret-token", "us-east-1")
        assert caught.value.failure is ProfileDiscoveryFailure.AUTH_REJECTED
        assert caught.value.auth_failure is True
        assert "secret-token" not in str(caught.value)
        assert "private-body" not in str(caught.value)

    @pytest.mark.asyncio
    async def test_auth_rejected_in_one_region_but_empty_in_another_is_not_found(self, install_transport) -> None:
        """
        What it does: Mixed results; at least one region answered successfully.
        Purpose: A successful empty answer is more informative than a 403.
        """
        install_transport(_by_region({"us-east-1": httpx.Response(403)}))

        with pytest.raises(ProfileDiscoveryError) as caught:
            await discover_profile_arn("tok", "us-east-1")
        assert caught.value.failure is ProfileDiscoveryFailure.NOT_FOUND

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectError("private-token"),
            httpx.ReadTimeout("private-token"),
            httpx.Response(500, text="private-token"),
            httpx.Response(429),
            httpx.Response(302, headers={"Location": "https://evil.invalid/"}),
            httpx.Response(200, text="not json private-token"),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={"profiles": {}}),
            httpx.Response(200, json={"profiles": [], "nextToken": 7}),
        ],
    )
    async def test_transport_and_malformed_failures_are_unavailable(self, install_transport, failure: object) -> None:
        """
        What it does: Every region fails with transport/HTTP/schema errors.
        Purpose: Classified as UNAVAILABLE so a configured ARN may be reused.
        """
        def handle(request: httpx.Request) -> httpx.Response:
            if isinstance(failure, Exception):
                raise failure
            return failure

        requests = install_transport(handle)

        with pytest.raises(ProfileDiscoveryError) as caught:
            await discover_profile_arn("tok", "us-east-1")
        assert caught.value.failure is ProfileDiscoveryFailure.UNAVAILABLE
        assert "private-token" not in str(caught.value)
        # Redirects are never followed, so the token cannot leave AWS hosts.
        assert all(r.url.host.endswith(".amazonaws.com") for r in requests)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("repeat", [True, False])
    async def test_pagination_is_bounded(self, install_transport, repeat: bool) -> None:
        """
        What it does: Infinite or repeating nextToken.
        Purpose: Broken pagination cannot hang gateway startup.
        """
        counter = {"n": 0}

        def handle(request: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            token = "same" if repeat else f"p{counter['n']}"
            return httpx.Response(200, json={"profiles": [{"arn": US_ARN}], "nextToken": token})

        requests = install_transport(handle)

        with pytest.raises(ProfileDiscoveryError):
            await discover_profile_arn("tok", "us-east-1", regions=["us-east-1"])
        assert len(requests) == (2 if repeat else PROFILE_LOOKUP_MAX_PAGES)


class TestLookupRegions:
    """Region ordering and validation."""

    def test_order_is_configured_then_api_then_kiro_regions_deduplicated(self) -> None:
        """
        What it does: Builds lookup order without duplicates.
        Purpose: Most likely region first; each region queried at most once.
        """
        order = profile_discovery._lookup_regions(EU_ARN, "ap-northeast-1")
        assert order == ["eu-central-1", "ap-northeast-1", "us-east-1"]

    @pytest.mark.parametrize("api_region", [None, "", "us-east-1.evil.invalid", "us--east-1", "US-EAST-1"])
    def test_invalid_api_region_is_dropped(self, api_region: object) -> None:
        """
        What it does: Ignores malformed regions.
        Purpose: Untrusted region text can never become a request host.
        """
        assert profile_discovery._lookup_regions(None, api_region) == list(KIRO_PROFILE_REGIONS)
