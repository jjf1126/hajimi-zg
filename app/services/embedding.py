import httpx
from typing import List, Union
from app.models.schemas import EmbeddingRequest, EmbeddingData, EmbeddingResponse, Usage
from app.utils.logging import log

class EmbeddingClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_name = request.model
        
        url = f"https://jjf1126-gemini-proxy.jjfanm.workers.dev/v1beta/models/{model_name}:batchEmbedContents"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input

        # The Gemini API expects a list of contents, so we format it this way.
        data = {
            "requests": [
                {
                    "model": f"models/{model_name}",
                    "content": {
                        "parts": [{"text": text}]
                    }
                } for text in inputs
            ]
        }
        
        extra_log = {
            "key": self.api_key[:8],
            "model": model_name,
        }
        log("INFO", "Embedding request started", extra=extra_log)

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            response_json = response.json()
            log("INFO", f"Google AI API response: {response_json}")
            
            # The response is a JSON object with an "embeddings" key.
            # Each item in the list is an object with a "values" key.
            embeddings = response_json.get("embeddings", [])

            embedding_data = [
                EmbeddingData(embedding=item["values"], index=i)
                for i, item in enumerate(embeddings)
            ]

            # The Gemini API does not provide token usage for embeddings.
            # We'll return a default usage object.
            usage = Usage(prompt_tokens=0, total_tokens=0)

            return EmbeddingResponse(
                object="list",
                data=embedding_data,
                model=model_name,
                usage=usage,
            )
