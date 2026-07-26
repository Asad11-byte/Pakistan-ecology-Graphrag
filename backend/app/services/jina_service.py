import httpx

from app.core.config import settings


class JinaService:
    """
    Wraps Jina's embeddings API. Used here purely for entity-linking:
    embedding entity names so a user's query can be matched to the
    closest graph node before traversal, via Neo4j's native vector index.
    """

    URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": settings.JINA_MODEL,
            "input": [{"text": t} for t in texts],
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.URL, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_one(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []


jina_service = JinaService()
