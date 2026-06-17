import logging
import os
import re
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from limits import enforce, get_user, limits_for, LimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yandex-proxy")

app = FastAPI()

YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net"
YANDEX_OPENAI_BASE = f"{YANDEX_BASE_URL}/v1"
YANDEX_IMAGES_URL = f"{YANDEX_OPENAI_BASE}/images/generations"

IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "aliceai-image-art-3.0")
_ALLOWED_SIZES = {"auto", "1x1", "1024x1024", "1536x1024", "1024x1536"}

_folder_id: Optional[str] = None


def _extract_folder_id(models_response: dict) -> Optional[str]:
    """Extract folder_id from model IDs like gpt://<folder_id>/yandex-gpt/latest"""
    for model in models_response.get("data", models_response.get("models", [])):
        model_id = model.get("id", "")
        match = re.match(r"^[a-z]+://([a-zA-Z0-9_-]+)/", model_id)
        if match:
            return match.group(1)
    return None


async def _get_folder_id(api_key: str) -> str:
    global _folder_id
    if _folder_id:
        return _folder_id

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{YANDEX_OPENAI_BASE}/models",
            headers={"Authorization": f"Api-Key {api_key}"},
        )
        if resp.status_code == 200:
            _folder_id = _extract_folder_id(resp.json())
            if _folder_id:
                logger.info(f"Extracted folder_id: {_folder_id}")
                return _folder_id

    raise ValueError("Could not extract folder_id from /v1/models response")


def _get_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return auth


def _log_user_headers(request: Request, endpoint: str):
    user_id = request.headers.get("x-openwebui-user-id")
    user_email = request.headers.get("x-openwebui-user-email")
    user_name = request.headers.get("x-openwebui-user-name")
    user_role = request.headers.get("x-openwebui-user-role")
    logger.info(
        f"[{endpoint}] user headers: "
        f"id={user_id!r} email={user_email!r} "
        f"name={user_name!r} role={user_role!r}"
    )


def _normalize_size(size: Optional[str]) -> str:
    if not size or size not in _ALLOWED_SIZES:
        return "1024x1024"
    return size


async def _generate_image_native(api_key: str, body: dict) -> dict:
    """Forward to the native OpenAI-compatible image endpoint, injecting model URI."""
    requested_model = body.get("model", "")
    if isinstance(requested_model, str) and requested_model.startswith("art://"):
        model = requested_model
    else:
        folder_id = await _get_folder_id(api_key)
        model = f"art://{folder_id}/{IMAGE_MODEL}"

    payload = {
        "prompt": body.get("prompt", ""),
        "model": model,
        "size": _normalize_size(body.get("size")),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(YANDEX_IMAGES_URL, json=payload, headers=headers)
        logger.info(f"Image request payload: {payload}")
        logger.info(f"Image response: {resp.status_code}")
        resp.raise_for_status()
        return resp.json()


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """Translate OpenAI image request to Yandex native OpenAI-compatible endpoint."""
    _log_user_headers(request, "images")

    user_id, role = get_user(request)
    daily, monthly = limits_for("image")
    try:
        await enforce(user_id, role, "image", daily, monthly)
    except LimitError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"message": e.message}},
        )

    api_key = _get_api_key(request)
    body = await request.json()

    try:
        result = await _generate_image_native(api_key, body)
        return JSONResponse(content=result)
    except httpx.HTTPStatusError as e:
        logger.error(f"Image generation failed: {e.response.status_code} {e.response.text}")
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": {"message": e.response.text}},
        )
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
)
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests to Yandex Cloud OpenAI-compatible API."""
    _log_user_headers(request, f"passthrough:{path}")

    if path.rstrip("/") == "v1/chat/completions":
        user_id, role = get_user(request)
        daily, monthly = limits_for("chat")
        try:
            await enforce(user_id, role, "chat", daily, monthly)
        except LimitError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": e.message}},
            )

    target_url = f"{YANDEX_BASE_URL}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

    # Cache folder_id from models response
    if path.rstrip("/") in ("v1/models", "models"):
        try:
            global _folder_id
            if not _folder_id:
                _folder_id = _extract_folder_id(resp.json())
                if _folder_id:
                    logger.info(f"Cached folder_id from models: {_folder_id}")
        except Exception:
            pass

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
