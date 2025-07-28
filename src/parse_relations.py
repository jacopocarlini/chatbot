from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434"  # fondamentale!
)

def extract_triples(text):
    prompt = f"""
Estrai relazioni dal testo nel formato (soggetto)-[relazione]->(oggetto).
Testo:
{text}
"""
    result = llm.invoke(prompt)
    lines = result.strip().split("\n")
    triples = []
    for line in lines:
        if "->" in line:
            parts = line.strip("()").split(")-[")
            if len(parts) == 2:
                subj = parts[0].strip()
                rel, obj = parts[1].split("]->(")
                triples.append((subj.strip(), rel.strip(), obj.strip(") \n").strip()))
    return triples