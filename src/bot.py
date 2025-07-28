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
    prompt = f"""
You are an assistant that generates Cypher queries for a graph where:
- All nodes are of type :Entity with a 'name' property
- All relationships are of type :RELATION with a 'type' property

Given a natural language question, return only the Cypher query that:
- Matches a triple in the form (subject)-[relationship]->(object)
- Uses the variable names: 'soggetto' for the subject node, 'relazione' for the relationship, and 'oggetto' for the object node
- Includes a WHERE clause only if needed to filter by 'name'
- Always returns: soggetto.name, relazione.type, oggetto.name

Do not include markdown, formatting, explanations, or quotes. Return only the raw Cypher query.

"{question}"
"""
    cypher = llm.invoke(prompt).strip().strip('`').strip("cypher")
    return {"cypher_query": cypher}

def extract_subject(state):
    question = state["question"]
    prompt = f"""
Utilizza solo una parola e non usare formattazione.
Estrai l'argomento principale della domanda:

"{question}"
"""
    cypher = llm.invoke(prompt).strip()
    # print('SUBJECT:', cypher)
    return {"cypher_query": cypher}


# --- 4. Nodo: esegui la query su Neo4j ---

def run_cypher(state):
    subject = state["cypher_query"]
    query = """
        MATCH (subject:Entity)-[rel:RELATION]->(object:Entity)
        WHERE toLower(subject.name) CONTAINS toLower($subject)
           OR toLower(object.name) CONTAINS toLower($subject)
        RETURN subject.name, rel.type, object.name
    """
    with driver.session() as session:
        result = session.run(query, subject=subject.lower())
        records = [dict(r) for r in result]
    # print(records)
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
    # print('PROMPT:', prompt)
    llama_answer = llm.invoke(prompt)
    answer = {"answer": llama_answer}
    return answer


# --- 6. Costruzione LangGraph ---

builder = StateGraph(GraphState)

builder.add_node("extract_subject", RunnableLambda(extract_subject))
builder.add_node("run_query", RunnableLambda(run_cypher))
builder.add_node("build_answer", RunnableLambda(build_answer))

builder.set_entry_point("extract_subject")
builder.add_edge("extract_subject", "run_query")
builder.add_edge("run_query", "build_answer")
builder.add_edge("build_answer", END)

graph_app = builder.compile()
