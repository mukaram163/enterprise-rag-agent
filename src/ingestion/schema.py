import psycopg2

def create_tables(db_connection_string: str, embedding_dim: int = 384):
    """
    Creates tables and vector extension safely without overwriting existing vector dimensions.
    """
    with psycopg2.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Safely check if document_chunks exists and verify vector dimension
            cur.execute("""
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute 
                WHERE attrelid = to_regclass('document_chunks') 
                AND attname = 'embedding';
            """)
            result = cur.fetchone()

            if result is None or result[0] is None:
                # Table does not exist -> create document_chunks
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({embedding_dim}),
                        allowed_users TEXT[] NOT NULL DEFAULT '{{}}',
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            else:
                # Dimension Safeguard: Verify expected dimension matches existing schema
                existing_type = result[0]
                expected_type = f"vector({embedding_dim})"
                if existing_type != expected_type:
                    raise ValueError(
                        f"Database vector dimension mismatch: Expected '{expected_type}', found '{existing_type}'."
                    )

            # Create feedback table non-destructively
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()