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
Debug logging module for requests.

Supports three modes (DEBUG_MODE):
- off: logging disabled
- errors: logs are saved only on errors (4xx, 5xx)
- all: logs are overwritten on each request

In "errors" mode, data is buffered in memory and flushed to files
only when flush_on_error() is called.

Also captures application logs (loguru) for each request and saves
them to app_logs.txt file for debugging convenience.

Request-scoped DebugSession (via contextvars) isolates concurrent
requests. An optional error-snapshot callback can forward full capture
bundles to an external reporter without touching the request path.
"""

from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from kiro.config import DEBUG_MODE, DEBUG_DIR


# Callback invoked after a local error flush with an immutable snapshot dict.
# Signature: (snapshot: dict) -> None. Must never raise into the request path.
ErrorSnapshotCallback = Callable[[dict[str, Any]], None]

_current_session: ContextVar[Optional["DebugSession"]] = ContextVar(
    "kiro_debug_session", default=None
)
_error_snapshot_callback: Optional[ErrorSnapshotCallback] = None


def set_error_snapshot_callback(callback: Optional[ErrorSnapshotCallback]) -> None:
    """Register/clear the global error-snapshot sink (tray / incident reporter)."""
    global _error_snapshot_callback
    _error_snapshot_callback = callback


def get_error_snapshot_callback() -> Optional[ErrorSnapshotCallback]:
    return _error_snapshot_callback


def get_current_session() -> Optional["DebugSession"]:
    return _current_session.get()


def classify_streaming_exception(exc: BaseException) -> tuple[str, str, str, int]:
    """Map a streaming-path exception to (source, code, phase, gateway_status)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "FirstToken" in name or "first token" in msg:
        return "network", "first_token_timeout", "first_token", 504
    if "Timeout" in name or "timeout" in msg:
        return "network", "timeout", "streaming", 504
    if (
        "RemoteProtocolError" in name
        or "incomplete chunked read" in msg
        or "incomplete message body" in msg
        or "peer closed connection" in msg
    ):
        # A response that breaks after streaming starts must not be replayed: the
        # client may already have consumed output. Classify it as an upstream
        # transport failure and let the route terminate the stream.
        return "network", "incomplete_upstream_response", "streaming", 502
    if "RequestError" in name or "ConnectError" in name or "connection" in msg:
        return "network", "connection_error", "connect", 502
    if "JSON" in name or "parse" in msg:
        return "gateway", "stream_parse_error", "response_parse", 500
    return "gateway", "streaming_error", "streaming", 500


@dataclass
class DebugSession:
    """Per-request capture buffers and metadata."""

    incident_id: str
    started_at: float
    path: str = ""
    model: str = ""
    stream: Optional[bool] = None
    request_body: Optional[bytes] = None
    kiro_request_body: Optional[bytes] = None
    raw_chunks: bytearray = field(default_factory=bytearray)
    modified_chunks: bytearray = field(default_factory=bytearray)
    app_logs: io.StringIO = field(default_factory=io.StringIO)
    loguru_sink_id: Optional[int] = None
    # Classification filled at flush time / by callers
    source: str = "unknown"
    code: str = "unknown"
    phase: str = "unknown"
    upstream_status: Optional[int] = None

    def clear_log_sink(self) -> None:
        if self.loguru_sink_id is not None:
            try:
                logger.remove(self.loguru_sink_id)
            except ValueError:
                pass
            self.loguru_sink_id = None

    def clear_buffers(self) -> None:
        self.request_body = None
        self.kiro_request_body = None
        self.raw_chunks = bytearray()
        self.modified_chunks = bytearray()
        self.clear_log_sink()
        self.app_logs = io.StringIO()

    def has_capture(self) -> bool:
        return bool(
            self.request_body
            or self.kiro_request_body
            or self.raw_chunks
            or self.modified_chunks
            or self.app_logs.getvalue().strip()
        )

    def setup_app_logs_capture(self) -> None:
        self.clear_log_sink()
        self.app_logs = io.StringIO()
        incident_id = self.incident_id

        def _filter(record: dict) -> bool:
            # Only capture logs emitted while this session is the active context.
            sess = _current_session.get()
            return sess is not None and sess.incident_id == incident_id

        self.loguru_sink_id = logger.add(
            self.app_logs,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            colorize=False,
            filter=_filter,
        )

    def build_snapshot(
        self,
        *,
        status_code: int,
        error_message: str,
        source: str,
        code: str,
        phase: str,
        upstream_status: Optional[int],
        client_disconnected: bool,
    ) -> dict[str, Any]:
        duration_ms = max(0, int((time.time() - self.started_at) * 1000))
        artifacts: dict[str, bytes] = {}
        if self.request_body:
            artifacts["request_body.json"] = self.request_body
        if self.kiro_request_body:
            artifacts["kiro_request_body.json"] = self.kiro_request_body
        if self.raw_chunks:
            artifacts["response_stream_raw.txt"] = bytes(self.raw_chunks)
        if self.modified_chunks:
            artifacts["response_stream_modified.txt"] = bytes(self.modified_chunks)
        app_logs = self.app_logs.getvalue()
        if app_logs.strip():
            artifacts["app_logs.txt"] = app_logs.encode("utf-8")
        error_info = {
            "status_code": status_code,
            "error_message": error_message,
            "source": source,
            "code": code,
            "phase": phase,
            "upstream_status": upstream_status,
            "client_disconnected": client_disconnected,
            "incident_id": self.incident_id,
            "path": self.path,
            "model": self.model,
            "stream": self.stream,
        }
        artifacts["error_info.json"] = json.dumps(
            error_info, indent=2, ensure_ascii=False
        ).encode("utf-8")

        artifact_meta = {
            name: {"bytes": len(data)} for name, data in artifacts.items()
        }
        return {
            "schema_version": 1,
            "kind": "kiro_gateway_incident",
            "incident_id": self.incident_id,
            "ts": int(self.started_at),
            "duration_ms": duration_ms,
            "path": self.path,
            "model": self.model or "unknown",
            "stream": self.stream,
            "status_code": status_code,
            "gateway_status": status_code,
            "upstream_status": upstream_status,
            "source": source,
            "code": code,
            "phase": phase,
            "client_disconnected": client_disconnected,
            "error_message": error_message,
            "artifacts": artifacts,
            "artifact_meta": artifact_meta,
        }


class DebugLogger:
    """
    Facade for managing debug request logs.

    Operating modes:
    - off: does nothing
    - errors: buffers data, flushes to files only on errors
    - all: writes data immediately to files (as before)

    Concurrent requests each get a DebugSession via contextvars. Local
    debug_dir remains last-write-wins (single folder), matching historical
    behaviour.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DebugLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.debug_dir = Path(DEBUG_DIR)
        self._initialized = True
        # Fallback session used by unit tests that call log_* without prepare.
        self._session: Optional[DebugSession] = None
        self._token = None

    # ---- property aliases so existing unit tests keep working ----

    def _active_session(self) -> Optional[DebugSession]:
        sess = _current_session.get()
        if sess is not None:
            return sess
        return self._session

    def _ensure_session(self) -> Optional[DebugSession]:
        if not self._is_enabled():
            return None
        sess = self._active_session()
        if sess is None:
            sess = DebugSession(
                incident_id=str(uuid.uuid4()),
                started_at=time.time(),
            )
            self._session = sess
            _current_session.set(sess)
        return sess

    def _clear_buffers(self):
        """Clears buffers for the session bound to the current context.

        Never touches another context's session via the instance fallback —
        that would break concurrent requests that share the DebugLogger
        singleton facade.
        """
        sess = _current_session.get()
        if sess is not None:
            sess.clear_buffers()
            try:
                _current_session.set(None)
            except Exception:
                pass
            if self._session is sess:
                self._session = None
            return
        if self._session is not None:
            self._session.clear_buffers()
            self._session = None

    def prepare_new_request(
        self,
        *,
        path: str = "",
        model: str = "",
        stream: Optional[bool] = None,
        incident_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Prepares the logger for a new request.

        In "all" mode: clears the logs folder.
        In "errors" mode: clears buffers.
        In both modes: sets up application log capture.

        Returns the new incident_id when logging is enabled, else None.
        """
        if not self._is_enabled():
            return None

        # Drop previous session for THIS context only (do not fall back to
        # self._session from another concurrent context).
        prev = _current_session.get()
        if prev is not None:
            prev.clear_buffers()
            try:
                _current_session.set(None)
            except Exception:
                pass
            if self._session is prev:
                self._session = None

        sess = DebugSession(
            incident_id=incident_id or str(uuid.uuid4()),
            started_at=time.time(),
            path=path or "",
            model=model or "",
            stream=stream,
        )
        self._session = sess
        _current_session.set(sess)
        sess.setup_app_logs_capture()

        if self._is_immediate_write():
            # "all" mode - clear folder and recreate (last-write-wins)
            try:
                if self.debug_dir.exists():
                    shutil.rmtree(self.debug_dir)
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"[DebugLogger] Directory {self.debug_dir} cleared for new request.")
            except Exception as e:
                logger.error(f"[DebugLogger] Error preparing directory: {e}")

        return sess.incident_id

    @property
    def _request_body_buffer(self) -> Optional[bytes]:
        sess = self._active_session()
        return sess.request_body if sess else None

    @_request_body_buffer.setter
    def _request_body_buffer(self, value: Optional[bytes]) -> None:
        sess = self._ensure_session()
        if sess:
            sess.request_body = value

    @property
    def _kiro_request_body_buffer(self) -> Optional[bytes]:
        sess = self._active_session()
        return sess.kiro_request_body if sess else None

    @_kiro_request_body_buffer.setter
    def _kiro_request_body_buffer(self, value: Optional[bytes]) -> None:
        sess = self._ensure_session()
        if sess:
            sess.kiro_request_body = value

    @property
    def _raw_chunks_buffer(self) -> bytearray:
        sess = self._active_session()
        return sess.raw_chunks if sess else bytearray()

    @property
    def _modified_chunks_buffer(self) -> bytearray:
        sess = self._active_session()
        return sess.modified_chunks if sess else bytearray()

    @property
    def _app_logs_buffer(self) -> io.StringIO:
        sess = self._active_session()
        return sess.app_logs if sess else io.StringIO()

    @property
    def _loguru_sink_id(self) -> Optional[int]:
        sess = self._active_session()
        return sess.loguru_sink_id if sess else None

    def _is_enabled(self) -> bool:
        """Checks if logging is enabled."""
        return DEBUG_MODE in ("errors", "all")

    def _is_immediate_write(self) -> bool:
        """Checks if immediate file writing is needed (all mode)."""
        return DEBUG_MODE == "all"

    def _clear_app_logs_buffer(self):
        """Clears the application logs buffer and removes sink."""
        sess = self._active_session()
        if sess is not None:
            sess.clear_log_sink()
            sess.app_logs = io.StringIO()

    def _setup_app_logs_capture(self):
        """Sets up application log capture to the active session buffer."""
        sess = self._ensure_session()
        if sess is None:
            return
        sess.setup_app_logs_capture()

    def update_request_meta(
        self,
        *,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        path: Optional[str] = None,
    ) -> None:
        """Fill in metadata discovered after prepare_new_request (e.g. parsed body)."""
        sess = self._active_session()
        if sess is None:
            return
        if model is not None:
            sess.model = model
        if stream is not None:
            sess.stream = stream
        if path is not None:
            sess.path = path

    def log_request_body(self, body: bytes):
        """
        Saves the request body (from client, OpenAI format).

        In "all" mode: writes immediately to file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        if self._is_immediate_write():
            self._write_request_body_to_file(body)
            sess = self._ensure_session()
            if sess:
                sess.request_body = body
        else:
            sess = self._ensure_session()
            if sess:
                sess.request_body = body

    def log_kiro_request_body(self, body: bytes):
        """
        Saves the modified request body (to Kiro API).

        In "all" mode: writes immediately to file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        if self._is_immediate_write():
            self._write_kiro_request_body_to_file(body)
            sess = self._ensure_session()
            if sess:
                sess.kiro_request_body = body
        else:
            sess = self._ensure_session()
            if sess:
                sess.kiro_request_body = body

    def log_raw_chunk(self, chunk: bytes):
        """
        Appends raw response chunk (from provider).

        In "all" mode: writes immediately to file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        if self._is_immediate_write():
            self._append_raw_chunk_to_file(chunk)
            sess = self._ensure_session()
            if sess:
                sess.raw_chunks.extend(chunk)
        else:
            sess = self._ensure_session()
            if sess:
                sess.raw_chunks.extend(chunk)

    def log_modified_chunk(self, chunk: bytes):
        """
        Appends modified chunk (to client).

        In "all" mode: writes immediately to file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        if self._is_immediate_write():
            self._append_modified_chunk_to_file(chunk)
            sess = self._ensure_session()
            if sess:
                sess.modified_chunks.extend(chunk)
        else:
            sess = self._ensure_session()
            if sess:
                sess.modified_chunks.extend(chunk)

    def log_error_info(self, status_code: int, error_message: str = ""):
        """
        Writes error information to file.

        Works in both modes (errors and all).
        In "all" mode writes immediately to file.
        In "errors" mode called from flush_on_error().

        Args:
            status_code: HTTP error status code
            error_message: Error message (optional)
        """
        if not self._is_enabled():
            return

        try:
            # Ensure directory exists
            self.debug_dir.mkdir(parents=True, exist_ok=True)

            sess = self._active_session()
            error_info = {
                "status_code": status_code,
                "error_message": error_message,
            }
            if sess is not None:
                error_info["incident_id"] = sess.incident_id
                error_info["path"] = sess.path
                error_info["model"] = sess.model
            error_file = self.debug_dir / "error_info.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)

            logger.debug(f"[DebugLogger] Error info saved (status={status_code})")
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing error_info: {e}")

    def flush_on_error(
        self,
        status_code: int,
        error_message: str = "",
        *,
        source: str = "unknown",
        code: str = "unknown",
        phase: str = "unknown",
        upstream_status: Optional[int] = None,
        client_disconnected: bool = False,
    ):
        """
        Flushes buffers to files on error and emits an immutable snapshot.

        In "errors" mode: flushes buffers and saves error_info.
        In "all" mode: only saves error_info (data already written) + snapshot.

        Classification kwargs:
          source: kiro_upstream | network | client_request | gateway | cancelled | unknown
          code:   machine-readable code (e.g. invalid_model, first_token_timeout)
          phase:  validation | connect | first_token | streaming | response_parse | unknown
        """
        if not self._is_enabled():
            return

        sess = self._active_session()
        if sess is None:
            # Nothing prepared; still write a minimal error_info for "all"/compat.
            if self._is_immediate_write():
                self.log_error_info(status_code, error_message)
            return

        # Infer defaults when callers pass the legacy 2-arg form.
        if source == "unknown":
            if client_disconnected:
                source = "cancelled"
            elif status_code == 422:
                source = "client_request"
            elif status_code in (502, 504):
                source = "network"
            elif 400 <= status_code < 600:
                source = "kiro_upstream"
        if code == "unknown":
            if client_disconnected:
                code = "client_disconnect"
            elif status_code == 422:
                code = "validation_error"
            elif status_code == 504:
                code = "timeout"
            elif status_code == 502:
                code = "bad_gateway"
            else:
                code = f"http_{status_code}"
        if phase == "unknown":
            if client_disconnected:
                phase = "streaming"
            elif status_code == 422:
                phase = "validation"
            else:
                phase = "unknown"
        if upstream_status is None and source == "kiro_upstream":
            upstream_status = status_code

        snapshot = sess.build_snapshot(
            status_code=status_code,
            error_message=error_message,
            source=source,
            code=code,
            phase=phase,
            upstream_status=upstream_status,
            client_disconnected=client_disconnected,
        )

        # In "all" mode data is already written, add error_info and app logs
        if self._is_immediate_write():
            self.log_error_info(status_code, error_message)
            self._write_app_logs_to_file()
            self._emit_snapshot(snapshot)
            self._clear_app_logs_buffer()
            self._clear_buffers()
            return

        try:
            # Create directory if not exists
            if self.debug_dir.exists():
                shutil.rmtree(self.debug_dir)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

            # Flush buffers to files
            if sess.request_body:
                self._write_request_body_to_file(sess.request_body)

            if sess.kiro_request_body:
                self._write_kiro_request_body_to_file(sess.kiro_request_body)

            if sess.raw_chunks:
                file_path = self.debug_dir / "response_stream_raw.txt"
                with open(file_path, "wb") as f:
                    f.write(sess.raw_chunks)

            if sess.modified_chunks:
                file_path = self.debug_dir / "response_stream_modified.txt"
                with open(file_path, "wb") as f:
                    f.write(sess.modified_chunks)

            # Save error information
            self.log_error_info(status_code, error_message)

            # Save application logs
            self._write_app_logs_to_file()

            logger.info(
                f"[DebugLogger] Error logs flushed to {self.debug_dir} "
                f"(status={status_code}, incident={sess.incident_id}, source={source})"
            )

        except Exception as e:
            logger.error(f"[DebugLogger] Error flushing buffers: {e}")
        finally:
            self._emit_snapshot(snapshot)
            # Clear buffers after flush
            self._clear_buffers()

    def flush_on_disconnect(self, message: str = "client disconnected") -> None:
        """Record a client disconnect as a cancelled incident (still uploads)."""
        self.flush_on_error(
            200,
            message,
            source="cancelled",
            code="client_disconnect",
            phase="streaming",
            client_disconnected=True,
        )

    def discard_buffers(self):
        """
        Clears buffers without writing to files.

        Called when request completed successfully in "errors" mode.
        Also called in "all" mode to save logs of successful request.
        """
        if DEBUG_MODE == "errors":
            self._clear_buffers()
        elif DEBUG_MODE == "all":
            # In "all" mode save logs even for successful requests
            self._write_app_logs_to_file()
            self._clear_app_logs_buffer()
            self._clear_buffers()

    def _emit_snapshot(self, snapshot: dict[str, Any]) -> None:
        cb = _error_snapshot_callback
        if cb is None:
            return
        try:
            # Pass a shallow copy; artifacts remain bytes (immutable enough).
            cb(snapshot)
        except Exception:
            # Never affect the request path.
            try:
                logger.debug("[DebugLogger] error snapshot callback failed", exc_info=True)
            except Exception:
                pass

    # ==================== Private file writing methods ====================

    def _write_request_body_to_file(self, body: bytes):
        """Writes request body to file."""
        try:
            file_path = self.debug_dir / "request_body.json"
            try:
                json_obj = json.loads(body)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(json_obj, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                with open(file_path, "wb") as f:
                    f.write(body)
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing request_body: {e}")

    def _write_kiro_request_body_to_file(self, body: bytes):
        """Writes Kiro request body to file."""
        try:
            file_path = self.debug_dir / "kiro_request_body.json"
            try:
                json_obj = json.loads(body)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(json_obj, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                with open(file_path, "wb") as f:
                    f.write(body)
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing kiro_request_body: {e}")

    def _append_raw_chunk_to_file(self, chunk: bytes):
        """Appends raw chunk to file."""
        try:
            file_path = self.debug_dir / "response_stream_raw.txt"
            with open(file_path, "ab") as f:
                f.write(chunk)
        except Exception:
            pass

    def _append_modified_chunk_to_file(self, chunk: bytes):
        """Appends modified chunk to file."""
        try:
            file_path = self.debug_dir / "response_stream_modified.txt"
            with open(file_path, "ab") as f:
                f.write(chunk)
        except Exception:
            pass

    def _write_app_logs_to_file(self):
        """Writes captured application logs to file."""
        try:
            sess = self._active_session()
            logs_content = sess.app_logs.getvalue() if sess else ""

            if not logs_content.strip():
                return

            # Ensure directory exists
            self.debug_dir.mkdir(parents=True, exist_ok=True)

            file_path = self.debug_dir / "app_logs.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(logs_content)

            logger.debug(f"[DebugLogger] App logs saved to {file_path}")
        except Exception:
            # Don't log error via logger to avoid recursion
            pass


# Global instance
debug_logger = DebugLogger()
