import sys
from extract import extract_text_from_pdf, extract_text_from_txt
from parse_relations import extract_triples
from neo4j_loader import insert_triple
from bot import graph_app

def main():
    # 1. Estrai testo
    path = sys.argv[1]
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(path)
    elif path.endswith(".txt"):
        text = extract_text_from_txt(path)
    else:
        print("File non supportato.")
        return

    # 2. Estrai triple
    triples = extract_triples(text)
    print("Triple trovate:", triples)

    # 3. Inserisci in Neo4j
    for subj, rel, obj in triples:
        insert_triple(subj, rel, obj)

    print("Dati inseriti in Neo4j.")

    # 4. Bot in loop
    print("\nScrivi la tua domanda. Digita 'exit' per uscire.")
    messages = []

    while True:
        user_input = input("\n🧑 Tu: ")
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = graph_app.invoke({"messages": messages})
        assistant_reply = response["messages"][-1]["content"]
        print("🤖 Bot:", assistant_reply)
        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
