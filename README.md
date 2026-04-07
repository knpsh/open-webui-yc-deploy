# open-webui-yc-deploy

This repository contains the code allowing to run open-webui with Yandex Cloud models, including YandexART (image generation) through a proxy and SpeechKit Recognition (speech-to-text) through a proxy.

1. Create a service account: https://yandex.cloud/en/docs/iam/operations/sa/create

2. Assign the roles for a folder: https://yandex.cloud/en/docs/iam/operations/sa/assign-role-for-sa

- `ai.imageGeneration.user`
- `ai.languageModels.user`
- `ai.models.user`
- `ai.speechkit-stt.user`
- `ai.speechkit-tts.user`
- `storage.editor`

3. Create a bucket: https://yandex.cloud/en/docs/storage/operations/buckets/create

4. Create a static key for the service account: https://yandex.cloud/en/docs/iam/operations/authentication/manage-access-keys#create-access-key

5. Create an API ket for the service account: https://yandex.cloud/en/docs/iam/operations/authentication/manage-api-keys#create-api-key

- `yc.ai.speechkitTts.execute`
- `yc.ai.speechkitStt.execute`
- `yc.ai.imageGeneration.execute`
- `yc.ai.foundationModels.execute`
- `yc.ai.languageModels.execute`

6. Fill the `docker-compose.yaml` file with:

- `<FOLDER_ID>` - folder id where roles for a service account were assigned;
- `<API_KEY>`
- `<S3_BUCKET>`
- `<S3_ACCESS_KEY>`
- `<S3_SECRET_KEY>`

7. Run `docker-compose up`

8. Open `http://localhost:3000`

9. Check available models, try image generation capability, try audio transcribation/summarization.
