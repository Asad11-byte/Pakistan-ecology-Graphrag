from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq

from app.core.config import settings
from app.services.jina_service import jina_service
from app.services.neo4j_service import neo4j_service

# ----------------------------------------------------------------------
# Predefined schema — constrains what the LLM is allowed to extract.
# Chosen to fit the Pakistan ecology domain specifically.
# ----------------------------------------------------------------------
ALLOWED_NODES = [
    "Ecosystem",
    "ProtectedArea",
    "Species",
    "Organization",
    "Threat",
    "Location",
    "Policy",
]

ALLOWED_RELATIONSHIPS = [
    "LOCATED_IN",
    "THREATENS",
    "PROTECTS",
    "MANAGES",
    "FEEDS_INTO",
    "PREYS_ON",
    "PARTNERS_WITH",
    "IMPLEMENTED_BY",
    "SUPPORTS",
    "HOSTS",
    "PART_OF",
]


def _get_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )


class GraphExtractionService:

    def __init__(self):
        self.llm = _get_llm()

        self.predefined_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            strict_mode=True,
            ignore_tool_usage=True,  # Groq/Llama tool-calling is unreliable for
                                      # structured extraction; use prompt-based
                                      # parsing instead, which is more robust.
        )

        self.inferred_transformer = LLMGraphTransformer(
            llm=self.llm,
            ignore_tool_usage=True,
            # No allowed_nodes / allowed_relationships — model infers freely
        )

    def extract(self, documents: list[dict], schema_mode: str) -> dict:
        """
        documents: list of {"id": ..., "title": ..., "text": ...}
        schema_mode: "predefined" | "llm_inferred"
        Returns extraction stats. Writes results into Neo4j tagged with schema_mode.
        """
        transformer = (
            self.predefined_transformer if schema_mode == "predefined" else self.inferred_transformer
        )

        lc_documents = [
            Document(page_content=doc["text"], metadata={"source_doc": doc["id"], "title": doc["title"]})
            for doc in documents
        ]

        node_names_seen = set()
        nodes_written = 0
        rels_written = 0
        failed_documents = []

        # Process one document at a time (not as a single batch call) so that
        # a malformed generation on one document doesn't abort the whole run.
        for lc_doc, source in zip(lc_documents, documents):
            source_id = source["id"]
            try:
                graph_documents = transformer.convert_to_graph_documents([lc_doc])
            except Exception as e:
                failed_documents.append({"id": source_id, "title": source["title"], "error": str(e)})
                continue

            for graph_doc in graph_documents:
                # ------------------------------------------------------
                # Batch-embed every NEW entity name in this document with
                # a single Jina API call, instead of one call per entity.
                # ------------------------------------------------------
                new_names = []
                for node in graph_doc.nodes:
                    if node.id not in node_names_seen:
                        new_names.append(node.id)
                        node_names_seen.add(node.id)

                embeddings_by_name = {}
                if new_names:
                    try:
                        vectors = jina_service.embed(new_names)
                        embeddings_by_name = dict(zip(new_names, vectors))
                    except Exception:
                        embeddings_by_name = {}  # fall back to no embeddings this batch

                for node in graph_doc.nodes:
                    neo4j_service.upsert_node(
                        label=node.type,
                        name=node.id,
                        schema_mode=schema_mode,
                        source_doc=source_id,
                        embedding=embeddings_by_name.get(node.id),
                    )
                    nodes_written += 1

                for rel in graph_doc.relationships:
                    try:
                        neo4j_service.upsert_relationship(
                            source_name=rel.source.id,
                            source_label=rel.source.type,
                            target_name=rel.target.id,
                            target_label=rel.target.type,
                            rel_type=rel.type,
                            schema_mode=schema_mode,
                        )
                        rels_written += 1
                    except Exception:
                        pass  # skip malformed individual relationships

        neo4j_service.ensure_vector_index()

        return {
            "schema_mode": schema_mode,
            "documents_processed": len(documents) - len(failed_documents),
            "documents_failed": len(failed_documents),
            "failed_documents": failed_documents,
            "node_writes": nodes_written,
            "relationship_writes": rels_written,
            "unique_entities_embedded": len(node_names_seen),
        }


graph_extraction_service = GraphExtractionService()