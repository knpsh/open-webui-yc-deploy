Running the open-webui for Yandex Cloud models support (with YandexART image generation proxy):

```bash
docker compose up -d --build
```

This starts:
- **yandex-proxy** (port 8081) — FastAPI proxy that translates OpenAI DALL-E image requests to YandexART and passes all other requests through to Yandex Cloud OpenAI-compatible API.
- **open-webui** (port 3000) — Open WebUI configured to use the proxy.

To enable image generation in Open WebUI (http://localhost:3000):
1. Go to **Admin Settings → Images**
2. Set image generation engine to **OpenAI DALL·E**
3. The API base URL and key are pre-configured via environment variables
4. Set **response format** to `b64_json` if there's an option
5. Enable image generation

YandexART docs: https://aistudio.yandex.ru/docs/ru/ai-studio/operations/generation/yandexart-request.html?tabs=programming_language_curl

  
