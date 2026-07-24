
"""
hybrid_retrieval.py - FIXED VERSION

Fixes:
1. BM25 initialization handles different metadata structures
2. Better error handling
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from bm25 import BM25
from llm import get_fast_llm, get_cached_llm_response
from cross_encoder import get_reranker

_GROQ_CONCURRENCY = 2
_BATCH_SIZE = 8
_BATCH_DELAY_SECONDS = 5.0


class HybridRetrieval:
    
    def __init__(self, vectorstore, embedding_loader):
        self.vector_store = vectorstore
        self.embedding_loader = embedding_loader
        self.bm25 = BM25()
        self.fast_llm = get_fast_llm()
        
        # Cross-encoder reranker
        self.cross_encoder = None
        try:
            self.cross_encoder = get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✓ Cross-encoder reranker loaded")
        except Exception as e:
            print(f"⚠ Cross-encoder not available: {e}")
        
        self._groq_semaphore = asyncio.Semaphore(_GROQ_CONCURRENCY)
        self._initialize_bm25_from_vectorstore()
    
    def _initialize_bm25_from_vectorstore(self):
        """FIXED: Load existing documents into BM25 index with better error handling"""
        print("[BM25 SYNC START] Initializing BM25 index from vector store collection...")
        try:
            all_data = self.vector_store.collection.get()
            
            if not all_data:
                print("[BM25 SYNC] No data returned from vector store")
                return
            
            # Check if we have documents
            documents_list = all_data.get('documents')
            if not documents_list:
                print("[BM25 SYNC] No documents found in vector store")
                return
            
            # Check if we have metadata
            metadatas_list = all_data.get('metadatas')
            
            # Build Document objects
            documents = []
            if metadatas_list and len(metadatas_list) == len(documents_list):
                # We have metadata - use it
                for doc_text, metadata in zip(documents_list, metadatas_list):
                    # Ensure metadata is a dict
                    if metadata is None:
                        metadata = {}
                    documents.append(Document(page_content=doc_text, metadata=metadata))
            else:
                # No metadata or mismatch - create documents without metadata
                print("⚠ Warning: No metadata available, creating documents without metadata")
                for doc_text in documents_list:
                    documents.append(Document(page_content=doc_text, metadata={}))
            
            if documents:
                self.bm25.add_documents(documents)
                print(f"[BM25 SYNC SUCCESS] Synced {len(documents)} documents to BM25 index.")
            else:
                print("⚠ No documents to add to BM25")
                
        except Exception as e:
            print(f"[BM25 SYNC ERROR] BM25 initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"BM25 initialization failed: {e}") from e
    
    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================
    
    async def add_documents(self, documents, use_contextual=True, use_llm_context=True):
        """Add documents with optional contextualization"""
        if not use_contextual:
            return await self._add_documents_simple(documents)
        return await self._add_documents_with_context(documents, use_llm_context)
    
    async def _add_documents_simple(self, documents):
        """Add documents without contextualization"""
        texts = [d.page_content for d in documents]
        embeddings = self.embedding_loader.generate_embedding(texts)
        self.vector_store.add_documents(documents, embeddings)
        self.bm25.add_documents(documents)
        
        return {
            "total_chunks": len(documents),
            "sources": list(set(d.metadata.get("source", "unknown") for d in documents)),
            "contextualization_used": "None",
        }
    
    async def _add_documents_with_context(self, documents, use_llm):
        """Add documents with contextualization"""
        doc_groups = self._group_by_source(documents)
        contextualized_docs = []
        
        for source, chunks in doc_groups.items():
            print(f"Processing {len(chunks)} chunks from {source}")
            doc_context = await self._get_document_context(chunks, use_llm)
            
            for batch_start in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[batch_start:batch_start + _BATCH_SIZE]
                batch_num = batch_start // _BATCH_SIZE + 1
                total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE
                print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                
                tasks = [
                    self._contextualize_chunk_throttled(chunk, doc_context, use_llm)
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                contextualized_docs.extend(batch_results)
                
                if use_llm and batch_start + _BATCH_SIZE < len(chunks):
                    print(f"  Rate limit pause ({_BATCH_DELAY_SECONDS}s)...")
                    await asyncio.sleep(_BATCH_DELAY_SECONDS)
        
        if contextualized_docs:
            texts = [doc.page_content for doc in contextualized_docs]
            embeddings = self.embedding_loader.generate_embedding(texts)
            self.vector_store.add_documents(contextualized_docs, embeddings)
            self.bm25.add_documents(contextualized_docs)
        
        return {
            "total_chunks": len(contextualized_docs),
            "sources": list(doc_groups.keys()),
            "contextualization_used": "LLM" if use_llm else "TEMPLATE",
        }
    
    def _group_by_source(self, documents):
        """Group documents by source"""
        groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in groups:
                groups[source] = []
            groups[source].append(doc)
        return groups
    
    async def _get_document_context(self, chunks, use_llm):
        """Generate document-level context"""
        if not use_llm:
            return f"Document: {chunks[0].metadata.get('source', 'unknown')}"
        
        sample = "\n\n".join(c.page_content[:200] for c in chunks[:3])
        prompt = f"""Analyze this document and provide a brief context (2-3 sentences):
1. Document type
2. Main topic
3. Key purpose

Document sample:
{sample}

Brief context:"""
        
        try:
            response = await get_cached_llm_response(self.fast_llm, prompt, use_cache=True)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"Context generation failed: {e}")
            return f"Document: {chunks[0].metadata.get('source', 'unknown')}"
    
    async def _contextualize_chunk_throttled(self, chunk, doc_context, use_llm):
        """Contextualize chunk with rate limiting"""
        async with self._groq_semaphore:
            result = await self._contextualize_chunk(chunk, doc_context, use_llm)
            if use_llm:
                await asyncio.sleep(0.5)
            return result
    
    async def _contextualize_chunk(self, chunk, doc_context, use_llm):
        """Add context to a chunk"""
        if not use_llm:
            contextual_content = f"{doc_context}\n\n{chunk.page_content}"
        else:
            prompt = f"""Add a brief 1-2 sentence introduction to situate this chunk.

Document context: {doc_context}

Chunk:
{chunk.page_content}

Provide introduction + original chunk.

Contextualized chunk:"""
            
            try:
                response = await get_cached_llm_response(self.fast_llm, prompt, use_cache=True)
                contextual_content = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"Chunk contextualization failed: {e}")
                contextual_content = f"{doc_context}\n\n{chunk.page_content}"
        
        new_metadata = chunk.metadata.copy()
        new_metadata["original_content"] = chunk.page_content
        return Document(page_content=contextual_content, metadata=new_metadata)
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    async def retrieve(self, query, top_k=5, use_hybrid=True, use_reranking=True, alpha=0.5, score_threshold=0.0):
        """Main retrieval with cross-encoder reranking"""
        print(f"[RETRIEVAL START] query='{query}', top_k={top_k}, hybrid={use_hybrid}, rerank={use_reranking}, threshold={score_threshold}")
        fetch_k = top_k * 3 if use_reranking else top_k
        
        # Get candidates
        if use_hybrid:
            results = self._hybrid_search(query, top_k=fetch_k, alpha=alpha)
        else:
            results = self._vector_search(query, top_k=fetch_k)
        
        if not results:
            print("[RETRIEVAL] No candidate documents found from search indexes.")
            return []
        
        score_key = "combined_score" if use_hybrid else "similarity_score"
        
        # Log the raw combined_score/similarity_score values for sanity check
        print(f"[RETRIEVAL CANDIDATES] Found {len(results)} raw candidates:")
        for idx, r in enumerate(results, 1):
            score_val = r.get(score_key, 0)
            meta = r.get("metadata", {}) or {}
            c_type = meta.get("content_type", "unknown")
            src = meta.get("source", "unknown")
            print(f"  Candidate {idx}: score={score_val:.4f} ({score_key}) | type={c_type} | src={src} | content_preview='{r['content'][:80]}...'")
        
        # Filter by score
        filtered_results = [r for r in results if r.get(score_key, 0) >= score_threshold]
        print(f"[RETRIEVAL FILTER] Score threshold {score_threshold} filtered out {len(results) - len(filtered_results)} candidates ({len(filtered_results)} remaining)")
        results = filtered_results
        
        # Cross-encoder reranking
        if use_reranking and len(results) > top_k and self.cross_encoder:
            print(f"[RETRIEVAL RERANK] Reranking {len(results)} candidates using CrossEncoder...")
            results = self.cross_encoder.rerank(
                query=query,
                documents=results,
                top_k=top_k,
                score_key="content"
            )
            print(f"[RETRIEVAL RERANK SUCCESS] Top candidate score: {results[0].get('final_score', results[0].get('combined_score', 0)):.4f}")
        
        final_results = results[:top_k]
        print(f"[RETRIEVAL SUCCESS] Returning {len(final_results)} document chunks.")
        return final_results
    
    def _vector_search(self, query, top_k):
        """Pure vector search"""
        query_embedding = self.embedding_loader.generate_embedding([query])[0]
        
        vector_results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        results = []
        if vector_results['documents'] and vector_results['documents'][0]:
            for doc_id, doc, metadata, distance in zip(
                vector_results['ids'][0],
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            ):
                results.append({
                    'id': doc_id,
                    'content': metadata.get('original_content', doc),
                    'contextualized_content': doc,
                    'metadata': metadata,
                    'similarity_score': round(1 - distance, 4),
                    'distance': round(distance, 4)
                })
        
        return results
    
    def _hybrid_search(self, query, top_k, alpha=0.5):
        """Hybrid vector + BM25 search"""
        query_embedding = self.embedding_loader.generate_embedding([query])[0]
        vector_results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        bm25_scores = self.bm25.get_scores(query)
        results = []
        
        if vector_results['documents'] and vector_results['documents'][0]:
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            
            for doc_id, doc, metadata, distance in zip(
                vector_results['ids'][0],
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            ):
                vector_score = 1 - distance
                doc_hash = hash(doc)
                bm25_score = bm25_scores.get(doc_hash, 0) / max_bm25 if max_bm25 > 0 else 0
                combined_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                results.append({
                    'id': doc_id,
                    'content': metadata.get('original_content', doc),
                    'contextualized_content': doc,
                    'metadata': metadata,
                    'vector_score': round(vector_score, 4),
                    'bm25_score': round(bm25_score, 4),
                    'combined_score': round(combined_score, 4),
                    'distance': round(distance, 4)
                })
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_stats(self):
        """Get system statistics"""
        try:
            return {
                "total_documents": self.vector_store.collection.count(),
                "collection_name": self.vector_store.collection_name,
                "bm25_documents": self.bm25.count(),
                "indices_synced": self.bm25.count() == self.vector_store.collection.count(),
                "cross_encoder_available": self.cross_encoder is not None
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "bm25_documents": 0,
                "indices_synced": False,
                "cross_encoder_available": False
            }
    
    def clear(self):
        """Clear all documents"""
        self.vector_store.clear_collection()
        self.bm25.clear()
        print("✓ All documents cleared")