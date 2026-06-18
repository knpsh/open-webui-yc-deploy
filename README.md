# open-webui-yc-deploy

This repository contains the code allowing to run open-webui with Yandex Cloud models, including YandexART (image generation) through a proxy and SpeechKit Recognition (speech-to-text) through a proxy.

## Components

- **open-webui** (`:3000`) — the chat UI. Talks to the proxies for chat, images, and speech-to-text.
- **yandex-proxy** (`:8081`) — OpenAI-compatible proxy for Yandex chat models (`/v1/chat/completions`) and YandexART image generation (`/v1/images/generations`).
- **stt-proxy** (`:8082`) — OpenAI-compatible proxy for SpeechKit speech-to-text (`/v1/audio/transcriptions`), using S3 for recognition.
- **redis** — stores per-user rate-limit counters.

## Per-user rate limiting

The proxies limit usage per user (using identity headers from Open WebUI), with daily and monthly windows stored in Redis.

- Admins bypass all limits.
- Chat token limits are counted from the model's response and include Open WebUI background calls (titles, tags, follow-ups).
- A blank or `0` limit means **unlimited** for that metric.
- `RATELIMIT_ENABLED` must be `true` for any limiting to apply.

## Environment variables (`.env`)

Copy `.env.example` to `.env` and fill in:

```
# Yandex Cloud
FOLDER_ID=
API_KEY=

# S3 (for speech-to-text)
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=

# OIDC / SSO (optional)
OIDC_CLIENT_ID=
OIDC_CLIENT_SECRET=
OIDC_DISCOVERY_URL=

# Rate limiting (blank or 0 = unlimited; RATELIMIT_ENABLED must be true to apply any limits)
RATELIMIT_ENABLED=true
RATELIMIT_CHAT_PER_DAY=
RATELIMIT_CHAT_PER_MONTH=
RATELIMIT_CHAT_TOKENS_PER_DAY=100000
RATELIMIT_CHAT_TOKENS_PER_MONTH=2000000
RATELIMIT_IMAGE_PER_DAY=20
RATELIMIT_IMAGE_PER_MONTH=300
RATELIMIT_STT_PER_DAY=50
RATELIMIT_STT_PER_MONTH=750
```

## Deployment

1. Create a service account: https://yandex.cloud/en/docs/iam/operations/sa/create

2. Assign the roles for a folder: https://yandex.cloud/en/docs/iam/operations/sa/assign-role-for-sa

- `ai.imageGeneration.user`
- `ai.languageModels.user`
- `ai.models.user`
- `ai.speechkit-stt.user`
- `ai.speechkit-tts.user`
- `storage.editor`
- `search-api.webSearch.user`

3. Create a bucket: https://yandex.cloud/en/docs/storage/operations/buckets/create

4. Create a static key for the service account: https://yandex.cloud/en/docs/iam/operations/authentication/manage-access-keys#create-access-key

5. Create an API ket for the service account: https://yandex.cloud/en/docs/iam/operations/authentication/manage-api-keys#create-api-key

- `yc.ai.speechkitTts.execute`
- `yc.ai.speechkitStt.execute`
- `yc.ai.imageGeneration.execute`
- `yc.ai.foundationModels.execute`
- `yc.ai.languageModels.execute`
- `yc.search-api.execute`

6. Fill in the `.env` file as describe above.

7. Run `docker-compose up -d --build`

8. Open `http://localhost:3000`

9. Check available models, try image generation capability, try audio transcribation/summarization, check limits.
