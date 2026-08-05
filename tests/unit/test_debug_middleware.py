# -*- coding: utf-8 -*-

"""
Unit tests for DebugLoggerMiddleware.
Tests debug logging initialization at the middleware level.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.requests import Request
from starlette.responses import Response


class TestDebugLoggerMiddlewareEndpointFiltering:
    """Tests for endpoint filtering in middleware."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/health", "/docs", "/"])
    async def test_skips_non_logged_endpoints(self, path: str):
        """
        What it does: Verifies that middleware skips endpoints not in LOGGED_ENDPOINTS.
        Purpose: Ensure health checks, docs, and root are not logged.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = path
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                response = await middleware.dispatch(mock_request, mock_call_next)
                
                mock_logger.prepare_new_request.assert_not_called()
                mock_call_next.assert_called_once_with(mock_request)
                assert response == mock_response
    
    @pytest.mark.asyncio
    async def test_processes_chat_completions_endpoint(self):
        """
        What it does: Verifies that middleware processes /v1/chat/completions.
        Purpose: Ensure OpenAI endpoint is logged.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/chat/completions"
            mock_request.body = AsyncMock(return_value=b'{"model": "test"}')
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                await middleware.dispatch(mock_request, mock_call_next)
                
                mock_logger.prepare_new_request.assert_called_once()
                mock_logger.log_request_body.assert_called_once_with(b'{"model": "test"}')
    
    @pytest.mark.asyncio
    async def test_processes_messages_endpoint(self):
        """
        What it does: Verifies that middleware processes /v1/messages.
        Purpose: Ensure Anthropic endpoint is logged.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/messages"
            mock_request.body = AsyncMock(return_value=b'{"model": "claude"}')
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                await middleware.dispatch(mock_request, mock_call_next)
                
                mock_logger.prepare_new_request.assert_called_once()


class TestDebugLoggerMiddlewareModeHandling:
    """Tests for DEBUG_MODE handling in middleware."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode,should_log", [
        ("off", False),
        ("errors", True),
        ("all", True),
    ])
    async def test_debug_mode_controls_logging(self, mode: str, should_log: bool):
        """
        What it does: Verifies that DEBUG_MODE controls whether requests are logged.
        Purpose: Ensure off disables logging, errors/all enable it.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', mode):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/chat/completions"
            mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                await middleware.dispatch(mock_request, mock_call_next)
                
                if should_log:
                    mock_logger.prepare_new_request.assert_called_once()
                else:
                    mock_logger.prepare_new_request.assert_not_called()
                
                mock_call_next.assert_called_once()


class TestDebugLoggerMiddlewareErrorHandling:
    """Tests for error handling in middleware."""
    
    @pytest.mark.asyncio
    async def test_handles_body_read_error_gracefully(self):
        """
        What it does: Verifies that middleware handles body read errors gracefully.
        Purpose: Ensure body read errors don't break the request.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/chat/completions"
            mock_request.body = AsyncMock(side_effect=Exception("Body read error"))
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                response = await middleware.dispatch(mock_request, mock_call_next)
                
                mock_logger.prepare_new_request.assert_called_once()
                mock_logger.log_request_body.assert_not_called()
                mock_call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_skips_empty_body(self):
        """
        What it does: Verifies that middleware doesn't log empty body.
        Purpose: Ensure empty requests don't create unnecessary logs.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/chat/completions"
            mock_request.body = AsyncMock(return_value=b'')
            
            mock_response = MagicMock(spec=Response)
            mock_call_next = AsyncMock(return_value=mock_response)
            
            with patch('kiro.debug_logger.debug_logger') as mock_logger:
                await middleware.dispatch(mock_request, mock_call_next)
                
                mock_logger.prepare_new_request.assert_called_once()
                mock_logger.log_request_body.assert_not_called()


class TestDebugLoggerMiddlewareResponsePassthrough:
    """Tests for transparent response passthrough."""
    
    @pytest.mark.asyncio
    async def test_returns_response_from_call_next(self):
        """
        What it does: Verifies that middleware returns response from call_next.
        Purpose: Ensure middleware doesn't modify the response.
        """
        with patch('kiro.debug_middleware.DEBUG_MODE', 'all'):
            from kiro.debug_middleware import DebugLoggerMiddleware
            
            middleware = DebugLoggerMiddleware(app=MagicMock())
            
            mock_request = MagicMock(spec=Request)
            mock_request.url.path = "/v1/chat/completions"
            mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
            
            expected_response = MagicMock(spec=Response)
            expected_response.status_code = 200
            mock_call_next = AsyncMock(return_value=expected_response)
            
            with patch('kiro.debug_logger.debug_logger'):
                actual_response = await middleware.dispatch(mock_request, mock_call_next)
                
                assert actual_response == expected_response
                assert actual_response.status_code == 200
