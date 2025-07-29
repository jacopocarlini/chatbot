import re
from typing import Tuple, List

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434"  # fondamentale!
)

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
puoi estrarre una lista di triple di stringhe [("concetto", "relazione", "concetto"), ...]  da questo testo

{text}
"""
)

def estrai_triplette(testo: str) -> List[Tuple[str, str, str]]:
    """
    Estrae una lista di triplette (soggetto, relazione, oggetto)
    da un testo che contiene righe nel formato:
    * [("concetto", "relazione", "concetto")]

    Args:
        testo (str): Testo di input.

    Returns:
        List[Tuple[str, str, str]]: Lista di triplette trovate.
    """
    pattern = re.compile(
        r'\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)'
    )

    triplette = []
    for match in pattern.finditer(testo):
        soggetto, relazione, oggetto = match.groups()
        # normalizzo rimuovendo spazi e convertendo "-" in stringa vuota
        triplette.append((
            soggetto.strip(),
            "" if relazione.strip() == "-" else relazione.strip(),
            oggetto.strip()
        ))
    return triplette

def extract_triples(text: str):
    prompt = prompt_template.format(text=text)
    result = llm.invoke(prompt)

    # Prova a valutare la risposta come lista Python
    try:
        triples = estrai_triplette(result)
        if isinstance(triples, list):
            return triples
    except Exception as e:
        print("⚠️ Errore durante il parsing delle triple:", e)
        print("Risposta del modello:", result)

    return []