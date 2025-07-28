from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(
    model="phi3:mini",
    base_url="http://localhost:11434"  # fondamentale!
)

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Dato il seguente testo, estrai tutte le entità e le relazioni significative nel formato di triplette (soggetto, relazione, oggetto).
Il soggetto e l'oggetto devono essere entità (es. persone, organizzazioni, luoghi, concetti).
La relazione deve descrivere come il soggetto e l'oggetto sono collegati.
Fornisci il risultato come una lista di triplette di stringhe. 

Esempio: [("soggetto", "relazione", "oggetto"), ...]

Testo da analizzare:
---
{text}
---
"""
)

def extract_triples(text: str):
    prompt = prompt_template.format(text=text)
    result = llm.invoke(prompt)

    # Prova a valutare la risposta come lista Python
    try:
        triples = eval(result)
        if isinstance(triples, list):
            return triples
    except Exception as e:
        print("⚠️ Errore durante il parsing delle triple:", e)
        print("Risposta del modello:", result)

    return []