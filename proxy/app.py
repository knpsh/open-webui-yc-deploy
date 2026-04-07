import asyncio
import logging
import re
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yandex-proxy")

app = FastAPI()

YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net"
YANDEX_OPENAI_BASE = f"{YANDEX_BASE_URL}/v1"
YANDEX_ART_URL = f"{YANDEX_BASE_URL}/foundationModels/v1/imageGenerationAsync"
YANDEX_OPERATION_URL = "https://operation.api.cloud.yandex.net/operations"

# Cached folder_id extracted from /v1/models
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


async def _generate_image(api_key: str, prompt: str, size: str = "1024x1024") -> str:
    """Call YandexART async API, poll for result, return base64 image."""
    folder_id = await _get_folder_id(api_key)
    model_uri = f"art://{folder_id}/yandex-art/latest"

    from math import gcd
    w_str, h_str = size.split("x") if "x" in size else ("1024", "1024")
    w, h = int(w_str), int(h_str)
    divisor = gcd(w, h)
    width = str(w // divisor)
    height = str(h // divisor)

    # YandexART has a 500 character prompt limit
    if len(prompt) > 500:
        prompt = prompt[:497] + "..."
        logger.warning("Prompt truncated to 500 characters")

    payload = {
        "modelUri": model_uri,
        "generationOptions": {
            "seed": "0",
            "aspectRatio": {
                "widthRatio": width,
                "heightRatio": height,
            },
        },
        "messages": [
            {
                "weight": "1",
                "text": prompt,
            }
        ],
    }

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(YANDEX_ART_URL, json=payload, headers=headers)
        logger.info(f"YandexART request payload: {payload}")
        logger.info(f"YandexART response: {resp.status_code} {resp.text}")
        resp.raise_for_status()
        operation = resp.json()
        operation_id = operation["id"]
        logger.info(f"YandexART operation started: {operation_id}")

        for _ in range(60):
            await asyncio.sleep(2)
            poll_resp = await client.get(
                f"{YANDEX_OPERATION_URL}/{operation_id}",
                headers=headers,
            )
            poll_resp.raise_for_status()
            result = poll_resp.json()

            if result.get("done"):
                if "error" in result:
                    raise RuntimeError(f"YandexART error: {result['error']}")
                image_base64 = result["response"]["image"]
                logger.info("YandexART generation complete")
                return image_base64

        raise TimeoutError("YandexART generation timed out")


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """Translate OpenAI DALL-E format to YandexART."""
    api_key = _get_api_key(request)
    body = await request.json()

    prompt = body.get("prompt", "")
    size = body.get("size", "1024x1024")
    n = body.get("n", 1)

    try:
        results = []
        for _ in range(n):
            b64_image = await _generate_image(api_key, prompt, size)
            results.append({"b64_json": b64_image})

        return JSONResponse(
            content={
                "created": 0,
                "data": results,
            }
        )
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
)
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests to Yandex Cloud OpenAI-compatible API."""
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
