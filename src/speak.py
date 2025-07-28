import sys
from extract import extract_text_from_pdf, extract_text_from_txt
from parse_relations import extract_triples
from neo4j_loader import insert_triple
from bot import graph_app


# Funzione per lanciare il bot
# Bot interattivo
def launch_bot():
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

def main():

    # 3. Avvia il bot
    launch_bot()

if __name__ == "__main__":
    main()
