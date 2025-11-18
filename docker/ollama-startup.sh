#!/bin/bash

# 1. Avvia il server Ollama in background e ottieni il suo PID
/usr/bin/ollama serve &
PID=$!

# 2. Attendi un momento per l'avvio del server (oppure fai un health check più robusto)
sleep 5

# 3. Scarica i modelli desiderati
echo "📥 Inizio download dei modelli..."
ollama pull phi3:mini
ollama pull llama3
ollama pull mistral
# gpt-oss:20b
echo "✅ Download modelli completato."

# 4. Attendi che il processo Ollama principale termini.
# Questo mantiene il container attivo.
wait $PID