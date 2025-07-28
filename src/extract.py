from PyPDF2 import PdfReader

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def extract_text_from_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()
