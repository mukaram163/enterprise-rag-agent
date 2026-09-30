from src.ingestion.loader import GoogleDriveLoader
from src.ingestion.splitter import TextSplitter
from src.retrieval.hybrid_pipeline import HybridRetriever
from src.generation.rag_chain import GeminiRAGGenerator

print("1. Ingesting Documents from Google Drive...")
loader = GoogleDriveLoader()
docs = loader.load_all_files(page_size=5)

if not docs:
    print("No supported files found in Google Drive.")
    exit()

print(f"Loaded {len(docs)} document(s) from Google Drive:")

for doc in docs:
    print(f" - {doc.metadata['source']} ({doc.metadata['mime_type']})")

print("\n2. Chunking Documents...")
splitter = TextSplitter()
all_chunks = []

for doc in docs:
    chunks = splitter.split_document(doc)
    all_chunks.extend(chunks)

print(f"Total chunks created across all documents: {len(all_chunks)}")

print("\n3. Initializing Hybrid Retriever...")
retriever = HybridRetriever(chunks=all_chunks)

query = "Explain the controlling function in management with an example."

print(f"\n4. Querying Hybrid Retriever with: '{query}'")

results = retriever.retrieve(
    query,
    top_k_retrieval=4,
    top_k_reranked=2
)

top_contexts = results["final_reranked"]

print(f"Retrieved {len(top_contexts)} top reranked chunk(s).")

print("\n5. Generating Response with Google Gemini...")
generator = GeminiRAGGenerator(model_name="gemini-3.8-flash")

answer = generator.generate_response(
    query=query,
    retrieved_items=top_contexts
)

print("\n=== GEMINI GENERATED RESPONSE ===")
print(answer)