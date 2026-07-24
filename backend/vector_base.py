"""
vector_base.py — ChromaDB vector store wrapper.
Fixed: self.collections → self.collection (consistent with rest of codebase)
"""
import hashlib
import os
from typing import List, Any
from pathlib import Path
import chromadb
import numpy as np


class VectorStore:
    def __init__(self, collection_name: str = "pdf_document", persist_directory: str = None):
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = str(Path.cwd() / "vector_store_v1")
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF documents for RAG", "hnsw:space": "cosine"},
            )
            print(f"VectorStore ready: {self.collection_name} ({self.collection.count()} docs)")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def clear_collection(self):
        try:
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            print("Collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def delete_document(self, doc_id: str):
        try:
            self.collection.delete(ids=[doc_id])
            print(f"Deleted: {doc_id}")
        except Exception as e:
            print(f"Error deleting {doc_id}: {e}")

    def delete_documents(self, doc_ids: List[str]):
        try:
            self.collection.delete(ids=doc_ids)
            print(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")

        existing_ids = set()
        try:
            existing = self.collection.get()
            if existing and existing["ids"]:
                existing_ids = set(existing["ids"])
        except:
            pass

        ids, metadatas, documents_text, embeddings_list = [], [], [], []
        seen_in_batch = set()

        for i, (doc, embed) in enumerate(zip(documents, embeddings)):
            src = doc.metadata.get("source", "unknown")
            raw_key = f"{src}_{i}_{doc.page_content}"
            doc_id = f"doc_{hashlib.md5(raw_key.encode()).hexdigest()[:16]}"
            if doc_id in existing_ids or doc_id in seen_in_batch:
                continue
            seen_in_batch.add(doc_id)
            ids.append(doc_id)
            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)
            documents_text.append(doc.page_content)
            embeddings_list.append(embed.tolist())

        if not ids:
            print(" [STORE DEDUP] No new documents (all duplicates)")
            return

        print(f"[STORE START] Adding {len(ids)} unique chunks to ChromaDB collection '{self.collection_name}'...")
        self.collection.upsert(
            ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=documents_text,
        )
        print(f"[STORE SUCCESS] Added {len(ids)} documents. Total in collection: {self.collection.count()}")


persist_dir = os.getenv("VECTOR_STORE_PATH", None)
vectorstore = VectorStore(persist_directory=persist_dir)