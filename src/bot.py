from typing import TypedDict, Optional, List

from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434"
)

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "verysecret"))


# --- 2. Stato iniziale ---

class GraphState(TypedDict, total=False):
    question: str
    cypher_query: str
    query_result: Optional[List[dict]]
    answer: str


# --- 3. Nodo: genera query Cypher ---

def generate_cypher_query(state):
    question = state["question"]
    keywords = [word.lower() for word in question.replace("?", "").split() if len(word) > 3]

    cypher = f"""
        UNWIND {keywords} AS kw
        MATCH (subject:Entity)-[rel:RELATION]->(object:Entity)
        WHERE toLower(subject.name) CONTAINS kw
        OR toLower(object.name) CONTAINS kw
        RETURN subject.name, rel.type, object.name
        """
    return {"cypher_query": cypher}


# --- 4. Nodo: esegui la query su Neo4j ---

def run_cypher(state):
    query = state["cypher_query"]

    with driver.session() as session:
        result = session.run(query)
        records = [dict(r) for r in result]
    # print('CONTEXT:', records)
    return {"query_result": records}


# --- 5. Nodo: costruisci risposta ---

def build_answer(state):
    question = state["question"]
    result = state["query_result"]

    if not result:
        return {"answer": "Non ho trovato informazioni utili nel grafo."}

    values = [
        (item['subject.name'], item['rel.type'], item['object.name'])
        for item in result
    ]

    prompt = f"Rispondi in italiano alla domanda '{question}' tenendo in considerazione esclusivamente le seguenti informazioni: {values}"
    print('PROMPT:', prompt)
    llama_answer = llm.invoke(prompt)
    answer = {"answer": llama_answer}
    return answer


# --- 6. Costruzione LangGraph ---

builder = StateGraph(GraphState)

builder.add_node("generate_cypher_query", RunnableLambda(generate_cypher_query))
builder.add_node("run_query", RunnableLambda(run_cypher))
builder.add_node("build_answer", RunnableLambda(build_answer))

builder.set_entry_point("generate_cypher_query")
builder.add_edge("generate_cypher_query", "run_query")
builder.add_edge("run_query", "build_answer")
builder.add_edge("build_answer", END)

graph_app = builder.compile()
