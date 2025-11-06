from typing import TypedDict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from config import graph, llm, embeddings, RAG_PROMPT


# --- 1. DEFINIZIONE DELLO STATO LANGGRAPH ---

class AgentState(TypedDict):
    """Rappresenta lo stato del grafo LangGraph."""
    question: str
    answer: str
    paragraphs: List[str]  # NUOVO: Lista dei paragrafi recuperati


# --- 2. STRUMENTI LANGCHAIN (TOOLS) CORRETTI ---


@tool("neo4j_retriever")
def retrieve_similar_question(question: str) -> List[str]:
    """
    Trova le 3 domande più simili nel database Neo4j,
    recupera i paragrafi collegati e li restituisce come lista di stringhe.
    """
    query_embedding = embeddings.embed_query(question)

    # Modifiche: LIMIT 3 e recupero del testo del Paragrafo collegato
    EMBEDDING_QUESTION_CYPHER = """
        CALL db.index.vector.queryNodes('embedding_question_index', 3, $query_embedding)
        YIELD node AS embedding_node, score
        MATCH (question_node:Question)-[:HAS_EMBEDDING]->(embedding_node)
        // Assumiamo che la domanda sia collegata alla sua risposta (Paragrafo)
        MATCH (question_node)-[:ANSWERED_BY]->(paragraph_node:Paragraph)
        RETURN paragraph_node.text AS paragraph_text, score
        ORDER BY score DESC
        LIMIT 3
        """

    try:
        results = graph.query(
            EMBEDDING_QUESTION_CYPHER,
            params={"query_embedding": query_embedding}
        )
    except Exception as e:
        print(f"Errore durante la query Cypher: {e}")
        return ["Errore di recupero dati."]

    print('   [Tool] Paragrafi trovati: ' , len(results))
    filtered_result = [
        item['paragraph_text']
        for item in results
        if item.get('score', 0.0) >= 0.8
    ]

    return filtered_result


# --- 3. NODI LANGGRAPH (SENZA MODIFICHE LOGICHE, SOLO CHIAMATE AI TOOL) ---

def retrieve(state: AgentState):
    """Nodo per il recupero del contesto RAG."""
    question = state["question"]
    print("\n-> 🧠 Passaggio 1: Recupero contesto da Neo4j...")

    # Chiama il tool modificato che restituisce List[str]
    paragraphs_list = retrieve_similar_question.invoke(question)

    # Unisce i paragrafi in una singola stringa per il 'context' RAG
    # Usiamo un separatore chiaro

    return {
        "paragraphs": paragraphs_list  # Salva la lista di stringhe nello stato
    }


def generate(state: AgentState):
    """Nodo per la generazione della risposta usando LLM."""
    question = state["question"]
    paragraphs = state["paragraphs"]

    print("-> 💬 Passaggio 2: Generazione della risposta...")

    # Catena di generazione
    chain = (
            ChatPromptTemplate.from_template(RAG_PROMPT)
            | llm
            | StrOutputParser()
    )

    # print(paragraphs)
    paragraphs_string = "\n---\n".join(paragraphs)

    answer = chain.invoke({
        "question": question,
        "paragraphs": paragraphs_string,
        "chat_history": ""
    })

    return {"answer": answer}
