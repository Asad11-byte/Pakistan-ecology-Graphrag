from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import settings
from app.services.jina_service import jina_service
from app.services.neo4j_service import neo4j_service

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions about Pakistan's ecology using "
     "a knowledge graph. You are given a subgraph of entities and the "
     "relationship chains connecting them to an anchor entity. Answer the "
     "question using ONLY this subgraph information. If the subgraph doesn't "
     "contain enough information, say so clearly."),
    ("human",
     "Question: {question}\n\n"
     "Anchor entity: {anchor}\n\n"
     "Connected entities and relationship paths:\n{subgraph}\n\n"
     "Answer:"),
])


class TraversalService:
    """
    Multi-hop traversal retrieval: anchor on an entity mentioned in the
    query (via Jina-embedding similarity against Neo4j's vector index,
    falling back to substring match), walk N hops of the graph, then
    ask Groq to synthesize an answer from the resulting subgraph.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0,
        )

    def _find_anchor(self, question: str, schema_mode: str) -> str | None:
        try:
            query_embedding = jina_service.embed_one(question)
            matches = neo4j_service.find_similar_entities(
                query_embedding, schema_mode=schema_mode, top_k=1
            )
            if matches:
                return matches[0]["name"]
        except Exception as e:
            print(f"[traversal] Vector-based anchor matching failed, falling back: {e}")

        # Fallback: word-overlap scoring instead of requiring an exact
        # substring match (which breaks on typos or partial phrasing,
        # e.g. "indus water dolphin" instead of "Indus River Dolphin").
        rows = neo4j_service.run(
            "MATCH (n {schema_mode: $mode}) RETURN n.name AS name",
            {"mode": schema_mode},
        )
        if not rows:
            return None

        STOPWORDS = {
            "how", "does", "do", "the", "a", "an", "is", "are", "in", "on",
            "of", "to", "and", "affect", "affects", "what", "why", "when",
            "where", "which", "this", "that", "with", "for", "impact",
        }
        question_words = {
            w.strip(".,?!") for w in question.lower().split() if w.strip(".,?!") not in STOPWORDS
        }

        best_name = None
        best_score = 0
        for row in rows:
            name = row["name"]
            if not name:
                continue
            name_words = {w.strip(".,?!") for w in name.lower().split()}
            overlap = len(question_words & name_words)
            if overlap > best_score:
                best_score = overlap
                best_name = name

        return best_name if best_name else rows[0]["name"]

    def answer(self, question: str, schema_mode: str = "predefined", hops: int = 3) -> dict:
        anchor = self._find_anchor(question, schema_mode)
        if not anchor:
            return {
                "answer": "No graph data found. Please ingest the dataset first.",
                "anchor": None,
                "subgraph": [],
            }

        results = neo4j_service.traverse(anchor, schema_mode=schema_mode, hops=hops)

        if not results:
            subgraph_text = f"(No connections found beyond the anchor entity '{anchor}'.)"
        else:
            lines = []
            for row in results:
                chain = " -> ".join(row["rel_chain"])
                lines.append(f"- {anchor} --[{chain}]--> {row['node']} (types: {row['labels']})")
            subgraph_text = "\n".join(lines)

        chain = ANSWER_PROMPT | self.llm
        response = chain.invoke({"question": question, "anchor": anchor, "subgraph": subgraph_text})

        return {
            "answer": response.content,
            "anchor": anchor,
            "subgraph": results,
        }


traversal_service = TraversalService()
