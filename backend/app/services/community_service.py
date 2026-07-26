import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import settings
from app.services.neo4j_service import neo4j_service

# In-memory cache of community summaries, keyed by schema_mode.
# Rebuilt whenever /api/graph/communities/rebuild is called (e.g. after
# ingestion). Avoids re-summarizing on every query (map-reduce is
# expensive to redo per-request).
_community_cache: dict[str, list[dict]] = {}

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarize a cluster of related entities from a knowledge graph "
     "about Pakistan's ecology into 2-3 sentences covering what connects "
     "them and why the cluster matters ecologically or administratively."),
    ("human", "Entities in this cluster and their relationships:\n{entities}\n\nSummary:"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer a broad/global question about Pakistan's ecology using "
     "a set of pre-written community summaries, each covering a cluster "
     "of related entities. Synthesize across all relevant summaries."),
    ("human", "Question: {question}\n\nCommunity summaries:\n{summaries}\n\nAnswer:"),
])


class CommunityService:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0,
        )

    def _build_networkx_graph(self, schema_mode: str) -> nx.Graph:
        nodes, edges = neo4j_service.get_graph(schema_mode)
        g = nx.Graph()
        for n in nodes:
            if n["name"]:
                g.add_node(n["name"], labels=n["labels"])
        for e in edges:
            if e["source"] and e["target"]:
                g.add_edge(e["source"], e["target"], rel_type=e["rel_type"])
        return g

    def rebuild_communities(self, schema_mode: str = "predefined") -> list[dict]:
        g = self._build_networkx_graph(schema_mode)

        if g.number_of_nodes() == 0:
            _community_cache[schema_mode] = []
            return []

        communities = nx.algorithms.community.louvain_communities(g, seed=42)

        summaries = []
        for idx, community_nodes in enumerate(communities):
            if len(community_nodes) < 2:
                continue  # skip trivial single-node "communities"

            subgraph = g.subgraph(community_nodes)
            entity_lines = []
            for u, v, data in subgraph.edges(data=True):
                entity_lines.append(f"{u} --[{data.get('rel_type', 'RELATED_TO')}]--> {v}")
            entities_text = "\n".join(entity_lines) if entity_lines else ", ".join(community_nodes)

            chain = MAP_PROMPT | self.llm
            response = chain.invoke({"entities": entities_text})

            summaries.append({
                "community_id": idx,
                "entities": sorted(community_nodes),
                "summary": response.content,
            })

        _community_cache[schema_mode] = summaries
        return summaries

    def answer(self, question: str, schema_mode: str = "predefined") -> dict:
        summaries = _community_cache.get(schema_mode)
        if summaries is None:
            summaries = self.rebuild_communities(schema_mode)

        if not summaries:
            return {
                "answer": "No community data available. Please ingest the dataset and rebuild communities first.",
                "communities_used": [],
            }

        summaries_text = "\n\n".join(
            f"Community {s['community_id']} ({', '.join(s['entities'][:6])}{'...' if len(s['entities']) > 6 else ''}):\n{s['summary']}"
            for s in summaries
        )

        chain = REDUCE_PROMPT | self.llm
        response = chain.invoke({"question": question, "summaries": summaries_text})

        return {
            "answer": response.content,
            "communities_used": [s["community_id"] for s in summaries],
            "community_summaries": summaries,
        }


community_service = CommunityService()
