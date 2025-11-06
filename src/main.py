from langgraph.graph import StateGraph, END

from agent_nodes import AgentState, retrieve, generate

def build_graph():
    """Costruisce e compila il grafo LangGraph."""
    print("\n--- Assemblaggio del Grafo LangGraph ---")

    workflow = StateGraph(AgentState)

    # Aggiungi i nodi (le funzioni definite in agent_nodes)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Definisci il punto di ingresso
    workflow.set_entry_point("retrieve")

    # Definisci le transizioni
    # Recupera -> Genera
    workflow.add_edge("retrieve", "generate")
    # Genera -> FINE
    workflow.add_edge("generate", END)

    # Compila il grafo
    app = workflow.compile()

    print("Grafo LangGraph compilato con successo.")
    return app

def run_chat_interface(app):
    """Loop principale per l'interazione da terminale."""
    print("\n=============================================")
    print("🤖 Agente di Chat Neo4j-Ollama (RAG + Memoria)")
    print("=============================================")
    print("Digita 'esci' per terminare. Digita 'mostra fonti' dopo una risposta per vedere i documenti originali.")

    show_context_next = False
    last_state: Dict[str, Any] = {}

    while True:
        try:
            user_input = input("\nUtente: ")

            if user_input.lower() in ["esci", "exit", "quit"]:
                print("Agente terminato. Arrivederci!")
                break

            # NUOVA LOGICA: mostra il contesto della risposta precedente
            if user_input.lower() == "mostra fonti":
                if last_state and last_state.get('paragraphs'):
                    print("\n--- 📖 Paragrafi di Contesto Usati ---")
                    for i, p in enumerate(last_state['paragraphs'], 1):
                        print(f"[{i}] {p}")
                    print("---------------------------------------")
                    continue
                else:
                    print("\nAgente: Nessun contesto precedente da mostrare o l'ultima azione non è stata una risposta.")
                    continue

            # Inizializza lo stato con l'input dell'utente
            initial_state = {
                "question": user_input,
                "paragraphs": [],
                "answer": ""
            }

            # Esegui il grafo
            final_state = app.invoke(initial_state)

            # Salva lo stato finale per il comando 'mostra fonti' successivo
            last_state = final_state

            # Stampa la risposta finale
            print(f"\nAgente: {final_state['answer']}")

            # *** Modifica Aggiunta: Suggerisci all'utente come vedere il contesto ***
            if final_state.get('paragraphs'):
                print("\n(Suggerimento: Digita 'mostra fonti' per vedere i paragrafi usati per questa risposta.)")

        except KeyboardInterrupt:
            print("\nAgente terminato (Interruzione Utente). Arrivederci!")
            break
        except Exception as e:
            print(f"\n!!! ERRORE CRITICO nell'esecuzione del grafo: {e}")

if __name__ == "__main__":

    # 1. Costruisci l'applicazione LangGraph
    langgraph_app = build_graph()

    # 2. Avvia l'interfaccia di chat
    run_chat_interface(langgraph_app)