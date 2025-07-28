import sys
import os
from extract import extract_text_from_pdf, extract_text_from_txt
from parse_relations import extract_triples
from neo4j_loader import insert_triple, clear_graph

# Funzione per elaborare un singolo file
def process_file(path):
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(path)
    elif path.endswith(".txt"):
        text = extract_text_from_txt(path)
    else:
        print(f"❌ File non supportato: {path}")
        return

    triples = extract_triples(text)

    for subj, rel, obj in triples:
        print(f"📤 Inserimento triple in Neo4j: ({subj} == {rel} => {obj})")
        insert_triple(subj.lower(), rel.lower(), obj.lower())

    print(f"Dati da {os.path.basename(path)} inseriti in Neo4j [{len(triples)}]")

# Funzione per elaborare tutti i file supportati in una cartella
def populate_graph_from_directory(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ '{folder_path}' non è una cartella valida.")
        return

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pdf") or f.endswith(".txt")
    ]

    if not files:
        print("⚠️ Nessun file .pdf o .txt trovato nella cartella.")
        return

    for file_path in files:
        process_file(file_path)

# Main
def main():
    if len(sys.argv) != 2:
        print("❗ Usa: python main.py <percorso-cartella>")
        return

    folder_path = sys.argv[1]
    # clear_graph()
    populate_graph_from_directory(folder_path)

if __name__ == "__main__":
    main()
