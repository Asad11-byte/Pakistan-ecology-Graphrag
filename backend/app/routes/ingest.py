import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.graph_extraction_service import graph_extraction_service
from app.services.neo4j_service import neo4j_service

router = APIRouter(prefix="/ingest", tags=["Ingest"])

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "pakistan_ecology_dataset.json"


def _load_documents() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


@router.post("/predefined")
async def ingest_predefined():
    """Run extraction using the predefined node/relationship schema."""
    documents = _load_documents()
    try:
        stats = graph_extraction_service.extract(documents, schema_mode="predefined")
        return {"status": "success", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predefined extraction failed: {str(e)}")


@router.post("/llm-inferred")
async def ingest_llm_inferred():
    """Run extraction letting the LLM infer its own schema (for comparison only)."""
    documents = _load_documents()
    try:
        stats = graph_extraction_service.extract(documents, schema_mode="llm_inferred")
        return {"status": "success", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM-inferred extraction failed: {str(e)}")


@router.post("/all")
async def ingest_all():
    """Run both extraction modes back-to-back. Recommended first call after deploy."""
    documents = _load_documents()
    try:
        predefined_stats = graph_extraction_service.extract(documents, schema_mode="predefined")
        inferred_stats = graph_extraction_service.extract(documents, schema_mode="llm_inferred")
        return {
            "status": "success",
            "predefined": predefined_stats,
            "llm_inferred": inferred_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/compare")
async def compare_schemas():
    """Compare stats between the two extraction strategies already in Neo4j."""
    try:
        predefined = neo4j_service.get_stats("predefined")
        inferred = neo4j_service.get_stats("llm_inferred")
        return {"predefined": predefined, "llm_inferred": inferred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.delete("/reset")
async def reset_graph():
    """Wipe all nodes/relationships — useful for re-running ingestion cleanly."""
    try:
        neo4j_service.run("MATCH (n) DETACH DELETE n")
        return {"status": "success", "message": "Graph cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
