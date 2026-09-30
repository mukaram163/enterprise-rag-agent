from typing import List
import uuid

from src.models.document import DocumentChunk


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """Recursively splits text into chunks under chunk_size limit."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = self.separators[-1]

        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + (
                len(separator) if current_chunk else 0
            )

            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    joined = separator.join(current_chunk).strip()

                    if joined:
                        chunks.append(joined)

                overlap_len = 0
                overlap_chunk = []

                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break

                current_chunk = overlap_chunk + [split]

                current_length = (
                    sum(len(x) for x in current_chunk)
                    + (len(separator) * (len(current_chunk) - 1))
                )

            else:
                current_chunk.append(split)
                current_length += split_len

        if current_chunk:
            final_joined = separator.join(current_chunk).strip()

            if final_joined:
                chunks.append(final_joined)

        return chunks

    def split_document(self, doc) -> List[DocumentChunk]:
        """Splits a LangChain Document into structured DocumentChunk objects."""

        # New loader returns a LangChain Document
        raw_chunks = self._split_text(doc.page_content)

        document_chunks = []

        # Metadata now comes from doc.metadata
        file_id = doc.metadata.get(
            "file_id",
            doc.metadata.get("gdrive_id", doc.metadata.get("source", "unknown"))
        )

        file_name = doc.metadata.get("source", "Document")

        page_number = doc.metadata.get("page_number", 1)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_obj = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                file_id=file_id,
                file_name=file_name,
                page_number=page_number,
                chunk_index=idx,
                text=chunk_text,
                metadata={
                    "total_chunks": len(raw_chunks),
                    **doc.metadata
                }
            )

            document_chunks.append(chunk_obj)

        return document_chunks