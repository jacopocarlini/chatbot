from typing import List
from uuid import uuid4

from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from neo4j import GraphDatabase
from config import graph, llm, embeddings, RAG_PROMPT

# --- CONFIGURAZIONE ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "verysecret"

# Inizializza Neo4j driver
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connessione a Neo4j stabilita con successo.")
except Exception as e:
    print(f"Errore di connessione a Neo4j: {e}")
    driver = None




# --- FUNZIONI DI BASE ---

def clear_graph():
    """Rimuove tutti i nodi e le relazioni dal grafo."""
    if driver:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Grafo Neo4j pulito.")

def clear_graph_and_drop_all_indexes():
    """
    Rimuove tutti i nodi, le relazioni, gli indici e i vincoli dal grafo Neo4j.
    """
    if not driver:
        print("Driver Neo4j non inizializzato.")
        return

    with driver.session() as session:
        # 1. Cancellazione di tutti i Nodi e Relazioni
        session.run("MATCH (n) DETACH DELETE n")
        print("Grafo Neo4j pulito (nodi e relazioni cancellati).")

        # 2. Cancellazione di tutti gli Indici e Vincoli

        # Query per ottenere tutti gli indici e i vincoli
        result = session.run("SHOW INDEXES")

        # Estrai i nomi degli indici e dei vincoli
        items_to_drop = []
        for record in result:
            item_name = record["name"]
            item_type = record["type"]

            # Neo4j 5 e superiori gestiscono tutti come indici, ma è utile distinguere per chiarezza
            if item_type in ["VECTOR", "RANGE", "TEXT", "LOOKUP", "FULLTEXT", "BTREE", "CONSTRAINT"]:
                # La sintassi per eliminare un vincolo è diversa da quella per eliminare un indice
                if item_type == "VECTOR" or item_type == "LOOKUP":
                    items_to_drop.append(f"INDEX `{item_name}`")
                else:
                    items_to_drop.append(f"CONSTRAINT `{item_name}`")


        # Esecuzione della cancellazione per ogni elemento
        if items_to_drop:
            print(f"Cancellazione di {len(items_to_drop)} indici/vincoli...")
            for item in items_to_drop:
                # Usa una transazione per eseguire le query di DROP
                try:
                    # La sintassi DROP INDEX/CONSTRAINT richiede il nome tra apici inversi (backticks)
                    session.run(f"DROP {item} IF EXISTS")
                    print(f"   [Cancellato] {item}")
                except Exception as e:
                    print(f"   [ERRORE] Errore nella cancellazione di {item}: {e}")
        else:
            print("Nessun indice o vincolo da cancellare trovato.")

    print("Operazione di pulizia completata.")


def create_indexes(tx):
    """Crea tutti gli indici necessari nel database Neo4j."""

    print("Creazione Indici Standard...")

    # 1. Indici per la ricerca veloce (Necessari per le clausole MATCH e MERGE)

    # Indice di unicità su Document.id (accelera MERGE)
    tx.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

    # Indice di unicità su Paragraph.id (accelera MERGE)
    tx.run("CREATE CONSTRAINT para_id_unique IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.id IS UNIQUE")

    # Indice di unicità su Question.id (accelera MERGE)
    tx.run("CREATE CONSTRAINT q_id_unique IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE")

    # Indice standard per la ricerca sul titolo
    tx.run("CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)")

    print("Indici Standard Creati.")

    # ---

    # 2. Indice Vettoriale (Essenziale per il RAG)
    print(f"Creazione Indice Vettoriale 'paragraph_embeddings' (Dimensione: 384)...")

    # Questo indice viene creato sui nodi :Embedding con la proprietà 'vector'.
    # È quello che serve alla query Cypher del tuo agent RAG.
    # **DEVI** assicurarti che la dimensione (dimensions) sia corretta.
    # SINTASSI CORRETTA PER NEO4J 4.4 / AURA LEGACY
    cypher_vector_index = f"""
    CREATE VECTOR INDEX embedding_question_index IF NOT EXISTS 
    FOR (e:EmbeddingQuestion) ON e.vector 
    OPTIONS {{
        indexProvider: 'vector-1.0',
        indexConfig: {{
            `vector.dimensions`: 384, 
            `vector.similarity_function`: 'cosine' 
        }}
    }}
    """
    tx.run(cypher_vector_index)
    print("Indice Vettoriale Creato.")

# Esempio di come chiamare la funzione (assumendo che tu abbia un driver Neo4j 'driver'):
# with driver.session() as session:
#     session.execute_write(create_indexes)

# --- FUNZIONI DI UTILITY (Dal tuo snippet) ---


def split_html_into_paragraphs_bs4(html_text: str) -> List[str]:
    """
    Divide il testo HTML in una lista di paragrafi
    utilizzando il parser Beautiful Soup.
    """
    # 1. Analizza l'HTML
    soup = BeautifulSoup(html_text, 'html.parser')

    # 2. Trova tutti i tag di paragrafo (<p>)
    paragraphs = soup.find_all('p')

    # 3. Estrai il testo da ciascun tag, rimuovendo gli spazi vuoti extra
    paragraph_texts = [
        p.get_text(strip=True)
        for p in paragraphs
        if p.get_text(strip=True)  # Filtra i paragrafi vuoti
    ]

    return paragraph_texts


# --- FUNZIONI DI TRANSAZIONE CYPER (private) ---

def _insert_doc_title_emb(tx, doc_id, title, embedding_title):
    tx.run("""
    MERGE (d:Document {id: $doc_id})
    ON CREATE SET d.title = $title
    MERGE (e:EmbeddingTitle {vector: $embedding_title})
    MERGE (d)-[:HAS_EMBEDDING]->(e)
    """, doc_id=doc_id, title=title, embedding_title=embedding_title)


def _insert_paragraph_and_emb(tx, doc_id, paragraph_id, text, embedding_paragraph):
    tx.run("""
    MATCH (d:Document {id: $doc_id})
    MERGE (p:Paragraph {id: $paragraph_id})
    ON CREATE SET p.text = $text
    MERGE (e:EmbeddingParagraph {vector: $embedding_paragraph})

    MERGE (d)-[:HAS_PARAGRAPH]->(p)
    MERGE (p)-[:HAS_EMBEDDING]->(e)
    """, doc_id=doc_id, paragraph_id=paragraph_id, text=text, embedding_paragraph=embedding_paragraph)


def _insert_paragraph_link(tx, source_paragraph_id, target_document_id):
    tx.run("""
    MATCH (p:Paragraph {id: $source_paragraph_id})
    MATCH (d:Document {id: $target_document_id})
    MERGE (p)-[:LINKS]->(d)
    """, source_paragraph_id=source_paragraph_id, target_document_id=target_document_id)


def _insert_question_and_answer(tx, question_id, text, embedding_question, answer_paragraph_id):
    tx.run("""
    MERGE (q:Question {id: $question_id})
    ON CREATE SET q.text = $text
    MERGE (e:EmbeddingQuestion {vector: $embedding_question})

    MERGE (q)-[:HAS_EMBEDDING]->(e)

    // Pass the merged Question node (q) and the parameter to the next step
    WITH q, $answer_paragraph_id AS answer_id 

    MATCH (p:Paragraph {id: answer_id})
    MERGE (q)-[:ANSWERED_BY]->(p)
    """, question_id=question_id, text=text, embedding_question=embedding_question,
           answer_paragraph_id=answer_paragraph_id)


# 🟢 --- FUNZIONI DI INSERIMENTO PUBBLICHE (CALCOLO EMBEDDING AUTOMATICO) ---

def insert_document_and_title_embedding(doc_id, title):
    """
    Calcola l'embedding del titolo (Ollama) e inserisce Documento ed Embedding.
    """
    if not driver: return
    embedding_title = embeddings.embed_query(title)
    if not embedding_title:
        print(f"ERRORE: Impossibile calcolare l'embedding per il titolo del doc ID: {doc_id}")
        return

    with driver.session() as session:
        session.execute_write(_insert_doc_title_emb, doc_id, title, embedding_title)


def insert_paragraph(doc_id, paragraph_id, text):
    """
    Calcola l'embedding del paragrafo (Ollama) e inserisce Paragrafo ed Embedding.
    """
    if not driver: return
    embedding_paragraph = embeddings.embed_query(text)
    if not embedding_paragraph:
        print(f"ERRORE: Impossibile calcolare l'embedding per il paragrafo ID: {paragraph_id}")
        return

    with driver.session() as session:
        session.execute_write(_insert_paragraph_and_emb, doc_id, paragraph_id, text, embedding_paragraph)


def insert_paragraph_link(source_paragraph_id, target_document_id):
    """
    Crea la relazione LINKS tra un Paragrafo e un Documento.
    """
    if not driver: return
    with driver.session() as session:
        session.execute_write(_insert_paragraph_link, source_paragraph_id, target_document_id)


def insert_question_and_answer(question_id, text, answer_paragraph_id):
    """
    Calcola l'embedding della domanda (Ollama) e crea Question, Embedding e relazione ANSWERED_BY.
    """
    if not driver: return
    embedding_question = embeddings.embed_query(text)
    if not embedding_question:
        print(f"ERRORE: Impossibile calcolare l'embedding per la domanda ID: {question_id}")
        return

    with driver.session() as session:
        session.execute_write(_insert_question_and_answer, question_id, text, embedding_question, answer_paragraph_id)


# --- ESEMPIO DI UTILIZZO (ADATTATO) ---

def create_graph_from_document():
    document_title = "Manuale Zucchetti TimeSheet Riepilogo"
    document_text = """
    <p>
    Questo manuale operativo descrive le procedure per la compilazione e l'invio del Timesheet e l'approvazione delle richieste. 
    Per compilare e inviare il Timesheet mensile in approvazione, il collaboratore deve accedere al modulo "ZTimesheet" e cliccare su "Compilazione WF Timesheet". 
    </p>
    <p>
    La deadline per l'invio del Timesheet è entro e non oltre il giorno 6 del mese successivo. 
    Le richieste devono essere approvate dal responsabile entro e non oltre il giorno 3 del mese successivo.
    </p>
    """
    doc_id = str(uuid4())
    print(f"Creazione Documento: {document_title} (ID: {doc_id})")

    # 1. Inserisce il Documento e il suo Title Embedding
    insert_document_and_title_embedding(doc_id, document_title)

    # 2. Suddivide il testo in paragrafi/chunk
    chunks = split_html_into_paragraphs_bs4(document_text)

    for i, chunk_text in enumerate(chunks):
        paragraph_id = f"{doc_id}_p{i + 1}"
        print(f"  - Inserimento Paragrafo {i + 1} (ID: {paragraph_id})")

        # Inserisce il Paragrafo e il suo Paragraph Embedding
        insert_paragraph(doc_id, paragraph_id, chunk_text)
        insert_paragraph_link(paragraph_id, doc_id)

    # Esempio di Domanda fittizia legata al primo paragrafo
    insert_question_and_answer(
        question_id=str(uuid4()),
        text="Qual è il processo per l'invio del Timesheet?",
        answer_paragraph_id=f"{doc_id}_p1"  # Assume che il primo chunk contenga la risposta
    )
    print("Grafo creato con successo.")


if __name__ == '__main__':

    clear_graph_and_drop_all_indexes()
    create_graph_from_document()

    with driver.session() as session:
        session.execute_write(create_indexes)

    # Non dimenticare di chiudere il driver alla fine
    if driver:
        driver.close()
