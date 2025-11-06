import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings # Scegliamo un embedding open-source standard
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURAZIONI ---
# Nota: Imposta queste variabili d'ambiente nel tuo terminale o sostituisci 'os.environ.get(...)'
# con i valori effettivi se preferisci non usare l'ambiente.

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "verysecret")

# Ollama e Modelli
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
# Questo modello DEVE essere lo stesso utilizzato per popolare gli embedding nel tuo grafo Neo4j.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --- INIZIALIZZAZIONE ---

# Connessione al Grafo
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    # Testa la connessione
    graph.query("RETURN 1")
    print(f"Connessione a Neo4j riuscita: {NEO4J_URI}")
except Exception as e:
    print(f"ERRORE: Impossibile connettersi a Neo4j. Assicurati che sia in esecuzione. Dettagli: {e}")
    exit()

# Modello LLM
llm = ChatOllama(model=OLLAMA_MODEL)
print(f"Modello Ollama inizializzato: {OLLAMA_MODEL}")

# Modello di Embedding
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"Modello di Embedding inizializzato: {EMBEDDING_MODEL}")


# --- PROMPT ---

RAG_PROMPT = """
Sei un assistente esperto e amichevole.
Usa il seguente contesto di recupero e la storia della chat per rispondere alla domanda.

STORIA DELLA CHAT:
{chat_history}

CONTESTO RECUPERATO:
{paragraphs}

DOMANDA: {question}

RISPOSTA:
"""