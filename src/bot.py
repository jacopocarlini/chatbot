from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import Tool, initialize_agent
from langchain_ollama import OllamaLLM
from neo4j import GraphDatabase
from typing import TypedDict, List

# 1. Definisci lo schema dello stato
class GraphState(TypedDict):
    messages: List[dict]

# 2. Connessione a Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "verysecret"))

def query_neo4j(cypher_query: str) -> str:
    with driver.session() as session:
        result = session.run(cypher_query)
        output = []
        for record in result:
            output.append(dict(record))
        return str(output) if output else "Nessun risultato trovato."

neo4j_tool = Tool(
    name="QueryNeo4j",
    func=query_neo4j,
    description="Esegui una query Cypher per cercare informazioni nel grafo Neo4j."
)

llm = OllamaLLM(model="llama3")

agent = initialize_agent(
    tools=[neo4j_tool],
    llm=llm,
    agent="chat-zero-shot-react-description",
    verbose=True
)

def answer_with_graph(state: GraphState) -> GraphState:
    question = state["messages"][-1]["content"]
    result = agent.run(question)
    return {
        "messages": add_messages(state["messages"], {"role": "assistant", "content": result})
    }

# 3. Aggiungi lo schema a StateGraph
builder = StateGraph(GraphState)
builder.add_node("search", answer_with_graph)
builder.set_entry_point("search")
builder.set_finish_point("search")

graph_app = builder.compile()
