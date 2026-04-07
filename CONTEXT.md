Running the open-webui for Yandex Cloud models support:

docker run -d \
  --name open-webui \
  -p 3000:8080 \
  --restart always \
  -v open-webui:/app/backend/data \
  -e ENABLE_OLLAMA_API=False \
  -e OPENAI_API_BASE_URL="https://ai.api.cloud.yandex.net/v1" \
  -e OPENAI_API_KEY="AQVNx2RrvxUcrYCy_MSusS2ltrebhBVHYkqxF8mG" \
  ghcr.io/open-webui/open-webui:main

  YandexART image generation:

  https://aistudio.yandex.ru/docs/ru/ai-studio/operations/generation/yandexart-request.html?tabs=programming_language_curl

  