from langchain_core.messages import HumanMessage

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
        # messages.append({"role": "user", "content": user_input})
        response = graph_app.invoke({"question": user_input})
        assistant_reply = response["answer"]
        print("🤖 Bot:", assistant_reply)
        # messages.append({"role": "assistant", "content": assistant_reply})


def main():
    # 3. Avvia il bot
    launch_bot()


if __name__ == "__main__":
    main()
