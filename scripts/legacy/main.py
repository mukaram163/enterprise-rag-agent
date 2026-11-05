import logging
from src.ingestion.schema import init_db
from src.ingestion.loader import GoogleDriveLoader
from src.ingestion.splitter import TextSplitter
from src.ingestion.indexer import PostgresVectorIndexer
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.rag_chain import GeminiRAGGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-main")

def run_end_to_end_demo():
    logger.info("1. Initializing PostgreSQL + pgvector Database Schema...")
    init_db()

    logger.info("2. Loading documents via GoogleDriveLoader...")
    loader = GoogleDriveLoader()
    docs = loader.load_all_files(page_size=5)
    logger.info(f"Loaded {len(docs)} document(s) from Google Drive.")

    if not docs:
        logger.warning("No documents returned by GoogleDriveLoader.")
        return

    logger.info("3. Chunking documents and indexing into PostgreSQL...")
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    indexer = PostgresVectorIndexer()

    total_chunks_indexed = 0
    for doc in docs:
        file_id = doc.metadata.get("file_id", "doc_unknown")
        file_name = doc.metadata.get("file_name", doc.metadata.get("source", "file_unknown"))
        text_len = len(doc.page_content) if hasattr(doc, "page_content") else 0
        
        chunks = splitter.split_document(doc)
        total_chunks_indexed += len(chunks)
        
        logger.info(f"  -> File: '{file_name}' ({file_id}) | Text length: {text_len} chars | Chunks created: {len(chunks)}")
        indexer.index_document(file_id=file_id, chunks=chunks)

    logger.info(f"Total chunks successfully processed and indexed: {total_chunks_indexed}")

    if total_chunks_indexed == 0:
        logger.error("0 chunks were generated. Pipeline stopped before retrieval.")
        return

    logger.info("4. Refreshing Sparse BM25 Retriever Index...")
    pipeline = RetrievalPipeline()
    pipeline.refresh()

    sample_query = "What is the summary of the assignment and orientation requirements?"
    test_user_id = "default_user"
    logger.info(f"5. Executing End-to-End Retrieval for Query: '{sample_query}'")

    retrieved_context = pipeline.retrieve(
        query=sample_query,
        candidate_k=9,
        top_k=3,
        user_id=test_user_id
    )

    logger.info(f"Retrieved {len(retrieved_context)} final reranked context chunk(s):")
    for idx, item in enumerate(retrieved_context, 1):
        logger.info(f"  [{idx}] {item['file_name']} (Page {item['page_number']}) - Rerank Score: {item.get('rerank_score', 0.0):.4f}")

    logger.info("6. Generating Answer with Gemini RAG Generator...")
    generator = GeminiRAGGenerator()
    answer = generator.generate_response(query=sample_query, retrieved_items=retrieved_context)

    print("\n" + "="*80)
    print("FINAL GEMINI RAG ANSWER:")
    print("="*80)
    print(answer)
    print("="*80 + "\n")

if __name__ == "__main__":
    run_end_to_end_demo()