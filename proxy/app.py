import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

import boto3
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from limits import (
    add_tokens,
    check_tokens,
    enforce,
    get_user,
    limits_for,
    LimitError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yandex-proxy")

app = FastAPI()

YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net"
YANDEX_OPENAI_BASE = f"{YANDEX_BASE_URL}/v1"
YANDEX_IMAGES_URL = f"{YANDEX_OPENAI_BASE}/images/generations"
YANDEX_RESPONSES_BASE_URL = (
    os.environ.get("YANDEX_RESPONSES_BASE_URL") or YANDEX_OPENAI_BASE
)

IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "aliceai-image-art-3.0")
FOLDER_ID = os.environ.get("FOLDER_ID", "")
CODE_INTERPRETER_MODEL_ID = (
    os.environ.get("YANDEX_CODE_INTERPRETER_MODEL_ID") or "yandex-code-interpreter"
)
CODE_INTERPRETER_MODEL = (
    os.environ.get("YANDEX_CODE_INTERPRETER_MODEL") or "qwen3-235b-a22b-fp8"
)
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")
CODE_INTERPRETER_OUTPUT_PREFIX = (
    os.environ.get("YANDEX_CODE_INTERPRETER_OUTPUT_PREFIX")
    or "code-interpreter-results"
).strip("/")
CODE_INTERPRETER_PRESIGN_EXPIRES = int(
    os.environ.get("YANDEX_CODE_INTERPRETER_PRESIGN_EXPIRES") or "604800"
)
CODE_INTERPRETER_INSTRUCTIONS = (
    os.environ.get("YANDEX_CODE_INTERPRETER_INSTRUCTIONS")
    or
    (
        "You are a Python programmer and can write and run code to solve the "
        "task you are given. First, check if you have the necessary libraries, "
        "and if not, install them. When you create output files, mention them "
        "in your final answer so they can be attached for the user."
    )
)
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


def _extract_total_tokens(resp: httpx.Response) -> int:
    """Extract usage.total_tokens from a buffered chat response.

    Handles both JSON (non-streaming) and SSE (streaming) bodies.
    """
    text = resp.text
    # Non-streaming JSON
    if not text.lstrip().startswith("data:"):
        try:
            return int(resp.json().get("usage", {}).get("total_tokens", 0) or 0)
        except Exception:
            return 0
    # Streaming SSE: scan data: lines for the last one carrying usage
    total = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            usage = json.loads(payload).get("usage")
            if usage and usage.get("total_tokens"):
                total = int(usage["total_tokens"])
        except Exception:
            continue
    return total


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


def _is_code_interpreter_model(model: object) -> bool:
    return isinstance(model, str) and model == CODE_INTERPRETER_MODEL_ID


def _messages_to_responses_input(messages: list[dict]) -> tuple[str, str]:
    system_parts = []
    conversation_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_to_text(message.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            conversation_parts.append(f"{role.upper()}:\n{content}")

    instructions = CODE_INTERPRETER_INSTRUCTIONS
    if system_parts:
        instructions = f"{instructions}\n\n" + "\n\n".join(system_parts)

    return instructions, "\n\n".join(conversation_parts)


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text"):
                    parts.append(str(item.get("text", "")))
                elif item.get("text"):
                    parts.append(str(item["text"]))
            elif item:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _safe_filename(filename: str) -> str:
    name = Path(filename or "generated-file").name.strip()
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    return name or "generated-file"


def _safe_path_part(value: Optional[str], fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("._-")
    return safe or fallback


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _upload_generated_file_to_s3(
    file_bytes: bytes,
    filename: str,
    user_id: Optional[str],
    response_id: Optional[str],
) -> tuple[str, str]:
    safe_user_id = _safe_path_part(user_id, "anonymous")
    safe_response_id = _safe_path_part(response_id, uuid.uuid4().hex)
    safe_filename = _safe_filename(filename)
    key = (
        f"{CODE_INTERPRETER_OUTPUT_PREFIX}/"
        f"{safe_user_id}/{safe_response_id}/{uuid.uuid4().hex}-{safe_filename}"
    )

    s3 = _get_s3_client()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=CODE_INTERPRETER_PRESIGN_EXPIRES,
    )
    return key, url


def _responses_payload(body: dict, folder_id: str, stream: bool) -> dict:
    instructions, input_text = _messages_to_responses_input(body.get("messages", []))
    payload = {
        "model": f"gpt://{folder_id}/{CODE_INTERPRETER_MODEL}",
        "input": input_text,
        "instructions": instructions,
        "tool_choice": "auto",
        "temperature": body.get("temperature", 0.3),
        "tools": [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                },
            }
        ],
        "stream": stream,
    }
    if body.get("max_tokens"):
        payload["max_output_tokens"] = body["max_tokens"]
    return payload


def _chat_completion(content: str, model: str) -> dict:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def _chat_chunk(chunk_id: str, model: str, content: str, done: bool = False) -> str:
    now = int(time.time())
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if done else {"content": content},
                "finish_reason": "stop" if done else None,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _iter_generated_file_refs(value: object) -> list[dict[str, str]]:
    refs = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            file_id = node.get("file_id")
            filename = node.get("filename") or node.get("name")
            if file_id and filename:
                refs.append(
                    {
                        "file_id": str(file_id),
                        "filename": _safe_filename(str(filename)),
                    }
                )
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)

    unique = {}
    for ref in refs:
        unique[(ref["file_id"], ref["filename"])] = ref
    return list(unique.values())


async def _download_generated_files(
    client: httpx.AsyncClient,
    api_key: str,
    folder_id: str,
    response: dict,
    user_id: Optional[str],
) -> tuple[list[dict[str, str]], list[str]]:
    refs = _iter_generated_file_refs(response)
    if not refs:
        return [], []

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
    }

    attachments = []
    failed = []
    response_id = response.get("id")
    for ref in refs:
        file_id = ref["file_id"]
        filename = ref["filename"]

        try:
            resp = await client.get(
                f"{YANDEX_RESPONSES_BASE_URL}/files/{file_id}/content",
                headers=headers,
            )
            resp.raise_for_status()
            key, url = _upload_generated_file_to_s3(
                resp.content,
                filename,
                user_id,
                response_id,
            )
        except Exception as e:
            logger.warning(f"Failed to attach generated file {file_id}: {e}")
            failed.append(filename)
            continue

        attachments.append(
            {
                "filename": filename,
                "file_id": file_id,
                "key": key,
                "url": url,
            }
        )

    return attachments, failed


def _attachments_markdown(
    attachments: list[dict[str, str]],
    failed: Optional[list[str]] = None,
) -> str:
    failed = failed or []
    if not attachments and not failed:
        return ""
    lines = ["", "", "Generated files:"]
    for attachment in attachments:
        filename = attachment["filename"]
        url = attachment["url"]
        lines.append(f"- [{filename}]({url})")
    for filename in failed:
        lines.append(f"- {filename} (could not attach)")
    return "\n".join(lines)


def _extract_response_text(response: dict) -> str:
    parts = []
    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        elif item_type == "code_interpreter_call":
            code = item.get("code")
            if code:
                parts.append(f"\n\n```python\n{code}\n```")
            for output in item.get("outputs", []) or []:
                logs = (output.get("logs") or "").strip()
                if logs:
                    parts.append(f"\n\n```\n{logs}\n```")
    return "".join(parts).strip()


async def _code_interpreter_completion(
    api_key: str,
    body: dict,
    user_id: Optional[str],
) -> JSONResponse:
    folder_id = FOLDER_ID or await _get_folder_id(api_key)
    payload = _responses_payload(body, folder_id, stream=False)
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
        "x-folder-id": folder_id,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{YANDEX_RESPONSES_BASE_URL}/responses",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        response_payload = resp.json()
        attachments, failed = await _download_generated_files(
            client, api_key, folder_id, response_payload, user_id
        )

    text = _extract_response_text(response_payload) + _attachments_markdown(
        attachments, failed
    )
    return JSONResponse(content=_chat_completion(text, CODE_INTERPRETER_MODEL_ID))


async def _code_interpreter_stream(
    api_key: str,
    body: dict,
    user_id: Optional[str],
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    sent_role = False
    response_id = None
    try:
        folder_id = FOLDER_ID or await _get_folder_id(api_key)
        payload = _responses_payload(body, folder_id, stream=True)
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
            "x-folder-id": folder_id,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{YANDEX_RESPONSES_BASE_URL}/responses",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    raw_event = line[len("data:"):].strip()
                    if not raw_event or raw_event == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw_event)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")
                    response = event.get("response") or {}
                    if response.get("id"):
                        response_id = response["id"]
                    elif event.get("response_id"):
                        response_id = event["response_id"]
                    content = ""
                    if event_type in (
                        "response.output_text.delta",
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    ):
                        content = event.get("delta", "")
                    elif event_type == "response.code_interpreter_call_code.done":
                        code = event.get("code", "")
                        if code:
                            content = f"\n\n```python\n{code}\n```\n\n"
                    elif event_type == "response.code_interpreter_call.in_progress":
                        content = "\n\n[Executing code...]\n\n"
                    elif event_type in (
                        "response.code_interpreter_call.done",
                        "response.code_interpreter_call.completed",
                    ):
                        content = "\n\n[Code executed]\n\n"

                    if content:
                        if not sent_role:
                            sent_role = True
                            yield _chat_chunk(chunk_id, CODE_INTERPRETER_MODEL_ID, "")
                        yield _chat_chunk(
                            chunk_id, CODE_INTERPRETER_MODEL_ID, content
                        )

            if response_id:
                response_resp = await client.get(
                    f"{YANDEX_RESPONSES_BASE_URL}/responses/{response_id}",
                    headers=headers,
                )
                response_resp.raise_for_status()
                attachments, failed = await _download_generated_files(
                    client, api_key, folder_id, response_resp.json(), user_id
                )
                attachments_text = _attachments_markdown(attachments, failed)
                if attachments_text:
                    yield _chat_chunk(
                        chunk_id, CODE_INTERPRETER_MODEL_ID, attachments_text
                    )
    except httpx.HTTPStatusError as e:
        message = f"Code interpreter failed: {e.response.status_code} {e.response.text}"
        logger.error(message)
        yield _chat_chunk(chunk_id, CODE_INTERPRETER_MODEL_ID, message)
    except Exception as e:
        message = f"Code interpreter failed: {e}"
        logger.error(message)
        yield _chat_chunk(chunk_id, CODE_INTERPRETER_MODEL_ID, message)

    yield _chat_chunk(chunk_id, CODE_INTERPRETER_MODEL_ID, "", done=True)
    yield "data: [DONE]\n\n"


async def _handle_code_interpreter_chat(
    request: Request,
    body: dict,
):
    api_key = _get_api_key(request)
    user_id, _ = get_user(request)
    try:
        if body.get("stream"):
            return StreamingResponse(
                _code_interpreter_stream(api_key, body, user_id),
                media_type="text/event-stream",
            )
        return await _code_interpreter_completion(api_key, body, user_id)
    except httpx.HTTPStatusError as e:
        logger.error(
            "Code interpreter failed: "
            f"{e.response.status_code} {e.response.text}"
        )
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": {"message": e.response.text}},
        )
    except Exception as e:
        logger.error(f"Code interpreter failed: {e}")
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


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

    is_chat = path.rstrip("/") == "v1/chat/completions"
    is_models = path.rstrip("/") in ("v1/models", "models")
    chat_user_id = chat_role = None
    parsed_body = None
    if is_chat:
        chat_user_id, chat_role = get_user(request)
        daily, monthly = limits_for("chat")
        t_daily, t_monthly = limits_for("chat_tokens")
        try:
            await enforce(chat_user_id, chat_role, "chat", daily, monthly)
            await check_tokens(
                chat_user_id, chat_role, "chat_tokens", t_daily, t_monthly
            )
        except LimitError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": e.message}},
            )

    target_url = f"{YANDEX_BASE_URL}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()
    if is_chat:
        try:
            parsed_body = json.loads(body)
            if _is_code_interpreter_model(parsed_body.get("model")):
                return await _handle_code_interpreter_chat(request, parsed_body)
            if parsed_body.get("stream"):
                opts = parsed_body.get("stream_options") or {}
                opts["include_usage"] = True
                parsed_body["stream_options"] = opts
                body = json.dumps(parsed_body).encode()
                headers.pop("content-length", None)
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

    if is_chat and resp.status_code == 200:
        logger.info(f"[chat-debug] raw resp tail: {resp.text[-600:]}")
        tokens = _extract_total_tokens(resp)
        logger.info(
            f"[chat-debug] user={chat_user_id} parsed_tokens={tokens} "
            f"body_len={len(resp.content)}"
        )
        if tokens:
            await add_tokens(chat_user_id, "chat_tokens", tokens)

    # Cache folder_id from models response and advertise our synthetic model.
    if is_models:
        try:
            global _folder_id
            models_payload = resp.json()
            if not _folder_id:
                _folder_id = _extract_folder_id(models_payload)
                if _folder_id:
                    logger.info(f"Cached folder_id from models: {_folder_id}")

            models_payload.setdefault("object", "list")
            models = models_payload.setdefault("data", models_payload.get("models", []))
            if isinstance(models, list) and not any(
                model.get("id") == CODE_INTERPRETER_MODEL_ID
                for model in models
                if isinstance(model, dict)
            ):
                models.append(
                    {
                        "id": CODE_INTERPRETER_MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "yandex-code-interpreter",
                    }
                )
            return JSONResponse(content=models_payload, status_code=resp.status_code)
        except Exception:
            pass

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
