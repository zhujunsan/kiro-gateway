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
Compat routes for OpenAI Responses / Codex wire_api extras.

Kept separate from ``routes_responses.py`` so store/CRUD wiring owned by
sibling work does not conflict:

- POST /v1/responses/compact — local history compaction
- GET  /v1/responses/models — Codex-shaped ``{models:[...]}`` list
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from kiro.codex_models import build_codex_models_list
from kiro.converters_compact import CompactRequest, build_compacted_response
from kiro.routes_openai import get_available_model_ids, verify_api_key

router = APIRouter(tags=["OpenAI Responses Compat"])



@router.post("/v1/responses/compact", dependencies=[Depends(verify_api_key)])
async def compact_response(request_data: CompactRequest):
    """
    Compact conversation history for the next ``/v1/responses`` call.

    Local approximation of OpenAI ``POST /responses/compact`` using history
    merge + byte-budget trim (AUTO_TRIM spirit). Does not call Kiro and does
    not use ``previous_response_id`` store chaining.
    """
    logger.info(f"Request to /v1/responses/compact (model={request_data.model})")

    if request_data.input is None:
        if request_data.previous_response_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "previous_response_id without input is not supported on "
                    "/v1/responses/compact; send the full input window to compact."
                ),
            )
        raise HTTPException(status_code=400, detail="input is required")

    try:
        body = build_compacted_response(
            model=request_data.model,
            input_data=request_data.input,
            instructions=request_data.instructions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info(
        "HTTP 200 - POST /v1/responses/compact - "
        f"items={body.get('metadata', {}).get('original_items')}->"
        f"{body.get('metadata', {}).get('final_items')} "
        f"trimmed={body.get('metadata', {}).get('compacted')}"
    )
    return JSONResponse(content=body)


@router.get("/v1/responses/models", dependencies=[Depends(verify_api_key)])
async def get_responses_models(request: Request):
    """
    Codex-shaped model list: ``{ "models": [ ModelInfo, ... ] }``.

    Prefer dual-compat ``GET /v1/models`` (also includes ``models``). This
    path exists for clients that probe Responses-specific discovery.
    """
    logger.info("Request to /v1/responses/models")
    model_ids = await get_available_model_ids(request)
    return JSONResponse(content={"models": build_codex_models_list(model_ids)})
