"""VectorStore for managing document embeddings"""
import hashlib
import os
from typing import List, Any
from pathlib import Path
import chromadb
import numpy as np


class VectorStore:
    """Manages documents embeddings in a ChromaDB vector store"""
    
    def __init__(
        self, 
        collection_name: str = "pdf_documents", 
        persist_directory: str = None
    ):
        """Initialize the Vector store"""
        self.collection_name = collection_name
        
        # FIXED: Use configurable path with sensible default
        if persist_directory is None:
            # Use absolute path relative to current working directory
            base_dir = Path.cwd()
            persist_directory = str(base_dir / "vector_store_v2")
        
        self.persist_directory = persist_directory
        self.collections = None
        self.client = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collections"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collections = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF documents embeddings for RAG",
                    "hnsw:space": "cosine"
                }
            )
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Persist directory: {self.persist_directory}")
            print(f"Existing documents in collection: {self.collections.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents in collection"""
        try:
            # Get all IDs first
            all_data = self.collections.get()
            if all_data['ids']:
                self.collections.delete(ids=all_data['ids'])
            print("Collection successfully cleared!")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def delete_document(self, doc_id: str):
        """Delete a specific document by ID"""
        try:
            self.collections.delete(ids=[doc_id])
            print(f"[SUCCESS] Deleted document: {doc_id}")
        except Exception as e:
            print(f"[ERROR] Error deleting document {doc_id}: {e}")
    
    def delete_documents(self, doc_ids: List[str]):
        """Delete specific documents by IDs"""
        try:
            self.collections.delete(ids=doc_ids)
            print(f"[SUCCESS] Deleted {len(doc_ids)} documents")
        except Exception as e:
            print(f"[ERROR] Error deleting documents: {e}")
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents with embeddings to the vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Get existing IDs to check for duplicates
        existing_ids = set()
        try:
            existing = self.collections.get()
            if existing and existing['ids']:
                existing_ids = set(existing['ids'])
        except:
            pass
        
        print(f"Adding {len(documents)} documents...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embed) in enumerate(zip(documents, embeddings)):
            # Generate consistent ID based on content hash
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            doc_id = f"doc_{content_hash}"
            
            # Skip if already exists
            if doc_id in existing_ids:
                continue

            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            embeddings_list.append(embed.tolist())
        
        if not ids:
            print("⚠️ No new documents (all duplicates)")
            return
        
        # Upsert documents
        self.collections.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )
        print(f"✓ Added/Updated {len(ids)} documents. Total: {self.collections.count()}")


# Create default vectorstore instance
# You can override persist_directory via environment variable
persist_dir = os.getenv("VECTOR_STORE_PATH", None)
vectorstore = VectorStore(persist_directory=persist_dir)