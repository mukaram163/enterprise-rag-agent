from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import TextSplitter
from src.retrieval.hybrid_pipeline import HybridRetriever
from src.generation.rag_chain import GeminiRAGGenerator

print("1. Loading and Indexing Document...")
doc = DocumentLoader().fetch_gdrive_file("1iniNVqmCoieRwR-DM7x7ubOkF0-vTquxS_K-NRoT9pg")
chunks = TextSplitter().split_document(doc)
retriever = HybridRetriever(chunks=chunks)

print("\n2. Querying Hybrid Retriever...")
query = "Explain the controlling function in management with an example."
results = retriever.retrieve(query, top_k_retrieval=4, top_k_reranked=2)

top_contexts = results["final_reranked"]
print(f"Retrieved {len(top_contexts)} top reranked chunks.")

print("\n3. Generating Response with Google Gemini...")
generator = GeminiRAGGenerator(model_name="gemini-3.8-flash")
answer = generator.generate_response(query=query, retrieved_items=top_contexts)

print("\n=== GEMINI GENERATED RESPONSE ===")
print(answer)
