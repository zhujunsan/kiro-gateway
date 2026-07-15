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
In-process TTL + LRU store for OpenAI Responses API objects.

Supports ``store`` / ``previous_response_id`` chaining and GET/DELETE
``/v1/responses/{id}``.

**Multi-instance limitation:** this store is process-local. Multiple gateway
replicas do **not** share state; ``previous_response_id`` chaining only works
within a single process. Not durable across restarts.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from kiro.config import RESPONSE_STORE_MAX_SIZE, RESPONSE_STORE_TTL


__all__ = [
    "StoredResponse",
    "ResponseStore",
    "normalize_input_items",
    "chain_input_with_previous",
    "should_store_response",
    "get_response_store",
    "reset_response_store",
]


@dataclass
class StoredResponse:
    """One stored Responses API turn (effective input + completed response)."""

    response_id: str
    response: Dict[str, Any]
    input: List[Any]
    created_at: float = field(default_factory=time.time)

    @property
    def output(self) -> List[Any]:
        out = self.response.get("output")
        return list(out) if isinstance(out, list) else []


def should_store_response(
    store: Optional[bool],
    previous_response_id: Optional[str] = None,
) -> bool:
    """
    OpenAI-ish default: store unless ``store`` is explicitly false.

    ``None`` (omitted) and ``True`` → store. ``False`` → do not store.

    ``previous_response_id`` does not change the decision (lookup is separate;
    chaining from a prior id works even when the new turn sets store=false).
    """
    _ = previous_response_id  # reserved for callers / future policy hooks
    return store is not False


def normalize_input_items(input_data: Union[str, List[Any], None]) -> List[Any]:
    """Normalize Responses ``input`` to a list of items for chaining/storage."""
    if input_data is None:
        return []
    if isinstance(input_data, str):
        return [{"type": "message", "role": "user", "content": input_data}]
    if isinstance(input_data, list):
        return list(input_data)
    return [{"type": "message", "role": "user", "content": str(input_data)}]


def chain_input_with_previous(
    previous: StoredResponse,
    new_input: Union[str, List[Any], None],
) -> List[Any]:
    """
    Build effective input for the next turn (OpenAI semantics):

    ``previous.input + previous.output + new_input``

    ``instructions`` are not inherited — callers resend them separately.
    """
    chained: List[Any] = []
    chained.extend(previous.input)
    chained.extend(previous.output)
    chained.extend(normalize_input_items(new_input))
    return chained


class ResponseStore:
    """
    Thread-safe in-process TTL LRU map of response_id → StoredResponse.

    Evicts expired entries on access and oldest entries when over max_size.
    """

    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        ttl = int(ttl_seconds if ttl_seconds is not None else RESPONSE_STORE_TTL)
        size = int(max_size if max_size is not None else RESPONSE_STORE_MAX_SIZE)
        if size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self._ttl = ttl
        self._max_size = size
        self._data: OrderedDict[str, StoredResponse] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def put(
        self,
        response_id: str,
        response: Dict[str, Any],
        input_items: Union[str, List[Any], None],
    ) -> StoredResponse:
        """Insert or replace a stored response (refreshes LRU position)."""
        if not response_id:
            raise ValueError("response_id is required")

        stored = StoredResponse(
            response_id=response_id,
            response=copy.deepcopy(response),
            input=normalize_input_items(input_items),
            created_at=time.time(),
        )
        with self._lock:
            if response_id in self._data:
                del self._data[response_id]
            self._data[response_id] = stored
            self._data.move_to_end(response_id)
            self._purge_expired()
            self._evict_overflow()
        return stored

    def get(self, response_id: str) -> Optional[StoredResponse]:
        """Return a deep copy of a stored response, or None if missing/expired."""
        with self._lock:
            self._purge_expired()
            stored = self._data.get(response_id)
            if stored is None:
                return None
            if self._is_expired(stored):
                del self._data[response_id]
                return None
            self._data.move_to_end(response_id)
            return StoredResponse(
                response_id=stored.response_id,
                response=copy.deepcopy(stored.response),
                input=copy.deepcopy(stored.input),
                created_at=stored.created_at,
            )

    def delete(self, response_id: str) -> bool:
        """Delete by id. Returns True if an entry was removed (and not expired)."""
        with self._lock:
            stored = self._data.get(response_id)
            if stored is None:
                return False
            if self._is_expired(stored):
                del self._data[response_id]
                return False
            del self._data[response_id]
            return True

    def _is_expired(self, stored: StoredResponse) -> bool:
        return (time.time() - stored.created_at) > self._ttl

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            rid
            for rid, stored in self._data.items()
            if (now - stored.created_at) > self._ttl
        ]
        for rid in expired:
            del self._data[rid]

    def _evict_overflow(self) -> None:
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)


_store: Optional[ResponseStore] = None
_store_lock = threading.Lock()


def get_response_store() -> ResponseStore:
    """Process-wide ResponseStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ResponseStore()
    return _store


def reset_response_store(
    ttl_seconds: Optional[int] = None,
    max_size: Optional[int] = None,
) -> ResponseStore:
    """Replace the singleton (for tests)."""
    global _store
    with _store_lock:
        _store = ResponseStore(ttl_seconds=ttl_seconds, max_size=max_size)
        return _store
