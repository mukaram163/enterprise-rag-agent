import os
import logging
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def get_db_config():
    return {
        "dbname": os.getenv("POSTGRES_DB", "enterprise_rag"),
        "user": os.getenv("POSTGRES_USER", "rag_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "rag_password"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
    }

def get_db_connection():
    """Returns a new PostgreSQL connection using the configured database settings."""
    return psycopg2.connect(**get_db_config())

def init_db():
    """Safely initializes or migrates the PostgreSQL database schema for pgvector."""
    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Check if table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'document_chunks'
        );
    """)
    table_exists = cur.fetchone()[0]

    if table_exists:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        row_count = cur.fetchone()[0]

        # Reliable detection of pgvector dimension via format_type
        cur.execute("""
            SELECT format_type(atttypid, atttypmod) 
            FROM pg_attribute 
            WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';
        """)
        dim_row = cur.fetchone()
        col_type = dim_row[0] if dim_row else ""

        if col_type != "vector(384)":
            if row_count == 0:
                logger.info(f"Dropping existing empty table with column type '{col_type}'. Recreating as vector(384)...")
                cur.execute("DROP TABLE document_chunks CASCADE;")
                table_exists = False
            else:
                cur.close()
                conn.close()
                raise RuntimeError(
                    f"Table document_chunks contains {row_count} rows with column type '{col_type}'. "
                    f"Explicit manual migration required to transition to vector(384)."
                )

    if not table_exists:
        cur.execute("""
            CREATE TABLE document_chunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                file_id VARCHAR(255) NOT NULL,
                file_name VARCHAR(255),
                text TEXT NOT NULL,
                page_number INT DEFAULT 1,
                chunk_index INT DEFAULT 0,
                embedding vector(384),
                allowed_users TEXT[] DEFAULT '{}',
                extra_metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        logger.info("Created document_chunks table with vector(384) and HNSW index.")

    conn.commit()
    cur.close()
    conn.close()
    logger.info("PostgreSQL schema initialized successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()