import asyncio
import json
import logging
import os
import uuid
from typing import Optional

import boto3
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-proxy")

app = FastAPI()

# Config from environment
FOLDER_ID = os.environ.get("FOLDER_ID", "")
API_KEY = os.environ.get("API_KEY", "")

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

STT_URL = "https://stt.api.cloud.yandex.net:443/stt/v3/recognizeFileAsync"
STT_RESULTS_URL = "https://stt.api.cloud.yandex.net:443/stt/v3/getRecognition"
OPERATION_URL = "https://operation.api.cloud.yandex.net/operations"

# Map content types to SpeechKit container types
CONTAINER_TYPE_MAP = {
    "audio/wav": "WAV",
    "audio/x-wav": "WAV",
    "audio/wave": "WAV",
    "audio/ogg": "OGG_OPUS",
    "audio/opus": "OGG_OPUS",
    "audio/webm": "OGG_OPUS",
    "audio/mp3": "MP3",
    "audio/mpeg": "MP3",
}


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _upload_to_s3(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Upload audio to S3, return (key, presigned_uri)."""
    s3 = _get_s3_client()
    key = f"stt-temp/{uuid.uuid4()}/{filename}"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)

    # Generate presigned URL so SpeechKit can access the file
    uri = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600,
    )
    logger.info(f"Uploaded to S3: {key}, presigned URL generated")
    return key, uri


def _delete_from_s3(key: str):
    """Clean up temp file from S3."""
    try:
        s3 = _get_s3_client()
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Deleted from S3: {key}")
    except Exception as e:
        logger.warning(f"Failed to delete S3 object {key}: {e}")


def _detect_container_type(content_type: str, filename: str) -> str:
    """Detect SpeechKit container type from content type or filename."""
    ct = (content_type or "").lower()
    if ct in CONTAINER_TYPE_MAP:
        return CONTAINER_TYPE_MAP[ct]

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ext_map = {
        "wav": "WAV",
        "ogg": "OGG_OPUS",
        "opus": "OGG_OPUS",
        "webm": "OGG_OPUS",
        "mp3": "MP3",
    }
    if ext in ext_map:
        return ext_map[ext]

    return "WAV"


async def _recognize_async(
    api_key: str, folder_id: str, s3_uri: str, container_type: str
) -> str:
    """Submit async recognition, poll for result, return text."""
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }

    payload = {
        "uri": s3_uri,
        "recognition_model": {
            "model": "general",
            "audio_format": {
                "container_audio": {
                    "container_audio_type": container_type,
                }
            },
            "language_restriction": {
                "restriction_type": "WHITELIST",
                "language_code": ["ru-RU", "en-US"],
            },
        },
    }

    async with httpx.AsyncClient(timeout=300) as client:
        # Submit async recognition
        resp = await client.post(STT_URL, json=payload, headers=headers)
        logger.info(f"STT submit response: {resp.status_code} {resp.text}")
        resp.raise_for_status()

        operation = resp.json()
        operation_id = operation["id"]
        logger.info(f"STT operation started: {operation_id}")

        # Poll operation until done
        for _ in range(120):
            await asyncio.sleep(2)
            poll_resp = await client.get(
                f"{OPERATION_URL}/{operation_id}",
                headers=headers,
            )
            poll_resp.raise_for_status()
            result = poll_resp.json()

            if result.get("done"):
                if "error" in result:
                    raise RuntimeError(f"STT error: {result['error']}")
                logger.info("STT operation done, fetching results")
                break
        else:
            raise TimeoutError("STT recognition timed out")

        # Fetch recognition results
        results_resp = await client.get(
            STT_RESULTS_URL,
            params={"operation_id": operation_id},
            headers=headers,
        )
        logger.info(f"STT results response: {results_resp.status_code}")
        results_resp.raise_for_status()

        # Parse results - response is NDJSON (multiple JSON objects)
        text_parts = []
        for line in results_resp.text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                result_data = chunk.get("result", {})
                # Prefer finalRefinement (normalized text)
                refinement = result_data.get("finalRefinement", {})
                if refinement:
                    normalized = refinement.get("normalizedText", {})
                    for alt in normalized.get("alternatives", []):
                        text = alt.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                elif result_data.get("final"):
                    for alt in result_data["final"].get("alternatives", []):
                        text = alt.get("text", "").strip()
                        if text:
                            text_parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to parse STT result chunk: {e}")

        full_text = " ".join(text_parts)
        logger.info(f"STT result: {full_text[:200]}...")
        return full_text


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
):
    """Whisper-compatible transcription endpoint."""
    s3_key = None
    try:
        file_bytes = await file.read()
        filename = file.filename or "audio.wav"
        content_type = file.content_type or ""

        logger.info(
            f"Received audio: {filename}, type={content_type}, size={len(file_bytes)}"
        )

        container_type = _detect_container_type(content_type, filename)
        logger.info(f"Detected container type: {container_type}")

        # Upload to S3
        s3_key, s3_uri = _upload_to_s3(file_bytes, filename)

        # Recognize
        text = await _recognize_async(API_KEY, FOLDER_ID, s3_uri, container_type)

        return JSONResponse(content={"text": text})

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e)}},
        )
    finally:
        if s3_key:
            _delete_from_s3(s3_key)


@app.get("/health")
async def health():
    return {"status": "ok"}
