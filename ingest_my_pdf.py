import pymupdf  # Modern PyMuPDF import
import requests
from src.ingestion.embeddings import EmbeddingGenerator

def ingest_pdf(file_path: str):
    doc = pymupdf.open(file_path)
    file_name = file_path.split("/")[-1]
    embedder = EmbeddingGenerator()
    
    chunks_payload = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            continue
            
        embeddings = embedder.generate_embeddings([text])
        
        chunks_payload.append({
            "id": f"{file_name}_p{page_num + 1}",
            "text": text,
            "embedding": embeddings[0],
            "metadata": {
                "source": file_name,
                "file_name": file_name,
                "page_number": page_num + 1
            }
        })

    payload = {
        "doc_id": file_name,
        "reindex": True,
        "chunks": chunks_payload
    }

    res = requests.post("http://localhost:8000/ingest", json=payload)
    print("Ingestion Result:", res.json())

if __name__ == "__main__":
    ingest_pdf("BSc-Hons-Medical-Laboratory-Technology-Curriculum.pdf")