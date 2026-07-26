from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.community_service import community_service
from app.services.traversal_service import traversal_service

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str
    schema_mode: str = "predefined"  # "predefined" | "llm_inferred"


@router.post("/traversal")
async def query_traversal(payload: QueryRequest):
    """Multi-hop traversal retrieval — best for specific, connective questions
    like 'How does glacial melt affect the Indus River Dolphin?'"""
    try:
        result = traversal_service.answer(payload.question, schema_mode=payload.schema_mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Traversal query failed: {str(e)}")


@router.post("/community")
async def query_community(payload: QueryRequest):
    """Community summarization retrieval — best for broad, global questions
    like 'What are the main environmental themes across Pakistan?'"""
    try:
        result = community_service.answer(payload.question, schema_mode=payload.schema_mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Community query failed: {str(e)}")


@router.post("/community/rebuild")
async def rebuild_communities(schema_mode: str = "predefined"):
    """Recompute Louvain communities + regenerate summaries. Call after ingestion."""
    try:
        summaries = community_service.rebuild_communities(schema_mode)
        return {"status": "success", "community_count": len(summaries), "communities": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Community rebuild failed: {str(e)}")
