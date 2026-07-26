from neo4j import GraphDatabase

from app.core.config import settings


class Neo4jService:
    """
    Thin wrapper around the Neo4j driver.

    Every node/relationship written by the graph-extraction step is
    tagged with a `schema_mode` property ("predefined" or "llm_inferred")
    so both extraction strategies can live in the same database without
    colliding, and retrieval can filter to just one mode.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict | None = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_node(self, label: str, name: str, schema_mode: str, source_doc: str, embedding: list[float] | None = None):
        # Every node also gets the shared :Entity label so the single
        # vector index (scoped to :Entity) can find it regardless of
        # its specific extracted type label.
        query = f"""
        MERGE (n:`{self._sanitize(label)}`:Entity {{name: $name, schema_mode: $schema_mode}})
        SET n.source_docs = coalesce(n.source_docs, []) + CASE
                WHEN $source_doc IN coalesce(n.source_docs, []) THEN []
                ELSE [$source_doc]
            END
        {"SET n.embedding = $embedding" if embedding is not None else ""}
        RETURN n
        """
        params = {"name": name, "schema_mode": schema_mode, "source_doc": source_doc}
        if embedding is not None:
            params["embedding"] = embedding
        self.run(query, params)

    def upsert_relationship(self, source_name: str, source_label: str, target_name: str,
                             target_label: str, rel_type: str, schema_mode: str):
        query = f"""
        MATCH (a:`{self._sanitize(source_label)}` {{name: $source_name, schema_mode: $schema_mode}})
        MATCH (b:`{self._sanitize(target_label)}` {{name: $target_name, schema_mode: $schema_mode}})
        MERGE (a)-[r:`{self._sanitize(rel_type)}` {{schema_mode: $schema_mode}}]->(b)
        RETURN r
        """
        self.run(query, {
            "source_name": source_name,
            "target_name": target_name,
            "schema_mode": schema_mode,
        })

    # ------------------------------------------------------------------
    # Vector index (for entity-linking via Jina embeddings)
    # ------------------------------------------------------------------

    def ensure_vector_index(self, dimensions: int = 1024):
        # One vector index across all node labels that have an `embedding` prop.
        query = """
        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
        FOR (n:Entity) ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: $dims,
            `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            self.run(query, {"dims": dimensions})
        except Exception:
            # Older AuraDB tiers may not support this syntax — traversal
            # falls back to fuzzy name matching if the index isn't present.
            pass

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_graph(self, schema_mode: str):
        """Return all nodes and relationships for a given schema_mode, for
        networkx-based community detection."""
        nodes = self.run(
            "MATCH (n) WHERE n.schema_mode = $mode RETURN n.name AS name, labels(n) AS labels",
            {"mode": schema_mode},
        )
        edges = self.run(
            """
            MATCH (a)-[r]->(b)
            WHERE r.schema_mode = $mode
            RETURN a.name AS source, b.name AS target, type(r) AS rel_type
            """,
            {"mode": schema_mode},
        )
        return nodes, edges

    def get_stats(self, schema_mode: str):
        node_count = self.run(
            "MATCH (n) WHERE n.schema_mode = $mode RETURN count(n) AS c", {"mode": schema_mode}
        )[0]["c"]
        rel_count = self.run(
            "MATCH ()-[r]->() WHERE r.schema_mode = $mode RETURN count(r) AS c", {"mode": schema_mode}
        )[0]["c"]
        node_labels = self.run(
            "MATCH (n) WHERE n.schema_mode = $mode RETURN DISTINCT labels(n) AS l", {"mode": schema_mode}
        )
        rel_types = self.run(
            "MATCH ()-[r]->() WHERE r.schema_mode = $mode RETURN DISTINCT type(r) AS t", {"mode": schema_mode}
        )
        return {
            "schema_mode": schema_mode,
            "node_count": node_count,
            "relationship_count": rel_count,
            "unique_node_labels": sorted({l for row in node_labels for l in row["l"]}),
            "unique_relationship_types": sorted({row["t"] for row in rel_types}),
        }

    def traverse(self, anchor_name: str, schema_mode: str, hops: int = 2, limit: int = 40):
        query = f"""
        MATCH (start {{schema_mode: $mode}})
        WHERE toLower(start.name) CONTAINS toLower($anchor)
        MATCH path = (start)-[*1..{hops}]-(connected)
        WHERE ALL(rel IN relationships(path) WHERE rel.schema_mode = $mode)
        WITH DISTINCT connected, start, path
        LIMIT $limit
        RETURN start.name AS anchor, connected.name AS node, labels(connected) AS labels,
               [r IN relationships(path) | type(r)] AS rel_chain
        """
        return self.run(query, {"anchor": anchor_name, "mode": schema_mode, "limit": limit})

    def find_similar_entities(self, embedding: list[float], schema_mode: str, top_k: int = 3):
        query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        WHERE node.schema_mode = $mode
        RETURN node.name AS name, labels(node) AS labels, score
        """
        try:
            return self.run(query, {"embedding": embedding, "top_k": top_k, "mode": schema_mode})
        except Exception:
            return []

    @staticmethod
    def _sanitize(label: str) -> str:
        # Neo4j labels/rel types can't contain backticks or spaces safely
        return "".join(ch for ch in label if ch.isalnum() or ch == "_") or "Entity"


neo4j_service = Neo4jService()
