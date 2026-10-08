import logging
from typing import List, Dict, Any, Optional, Union
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """Recursive character/separator-based splitting logic preserving original chunking semantics."""
        final_chunks = []
        
        # Determine the highest priority separator present in text
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = ""
                break
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                # Recursively split larger blocks using lower-priority separators
                if separator != "":
                    sub_splitter = TextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=[sep for sep in self.separators if sep != separator]
                    )
                    final_chunks.extend(sub_splitter._split_text(s))
                else:
                    final_chunks.append(s[:self.chunk_size])

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Combines split elements into chunk_size chunks respecting chunk_overlap."""
        docs = []
        current_doc = []
        total = 0

        for d in splits:
            len_d = len(d)
            if total + len_d + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc)
                    docs.append(doc_text)
                    
                    # Apply chunk_overlap by keeping trailing elements
                    while total > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total -= len(removed) + (len(separator) if current_doc else 0)
                
            current_doc.append(d)
            total += len_d + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))

        return docs

    def split_document(self, doc: Any) -> List[Union[DocumentChunk, Dict[str, Any]]]:
        """
        Splits input documents into chunks while preserving metadata on every chunk.
        Supports dictionary representations and custom Document objects.
        """
        # Read text from dict keys or object attributes
        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("page_content") or ""
            metadata = dict(doc.get("metadata", {}))
        else:
            text = (
                getattr(doc, "text", None)
                or getattr(doc, "page_content", None)
                or ""
            )
            raw_meta = getattr(doc, "metadata", {})
            metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}

        if not text or not text.strip():
            logger.warning("Document content is empty.")
            return []

        text_chunks = self._split_text(text)

        # Dictionary payload path (e.g. for simple dictionary inputs)
        if isinstance(doc, dict) and "file_id" not in doc and "file_id" not in metadata:
            chunks = []
            for chunk_text in text_chunks:
                chunks.append({
                    "text": chunk_text,
                    "metadata": dict(metadata)
                })
            return chunks

        # Structured DocumentChunk object path
        file_id = getattr(doc, "file_id", metadata.get("file_id", "doc_unknown"))
        file_name = getattr(doc, "file_name", metadata.get("file_name", metadata.get("source", "file_unknown")))
        page_number = getattr(doc, "page_number", metadata.get("page_number", 1))
        allowed_users = getattr(doc, "allowed_users", metadata.get("allowed_users", []))
        extra_metadata = getattr(doc, "extra_metadata", metadata.get("extra_metadata", metadata))

        chunks: List[Union[DocumentChunk, Dict[str, Any]]] = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{file_id}_p{page_number}_c{chunk_idx}"

            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                file_id=file_id,
                file_name=file_name,
                page_number=page_number,
                text=chunk_text,
                chunk_index=chunk_idx,
                allowed_users=allowed_users,
                extra_metadata=dict(extra_metadata)
            )

            # Enable dot-attribute access for .metadata compatibility (e.g., chunk.metadata)
            if not hasattr(chunk_obj, "metadata"):
                setattr(chunk_obj, "metadata", dict(extra_metadata))

            chunks.append(chunk_obj)

        return chunks