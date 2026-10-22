"""
hybrid_search.py

Hybrid BM25 + Vector Retrieval with optional Cross-Encoder Reranking and structured logging.
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from bm25 import BM25
from llm import get_fast_llm, get_cached_llm_response
from cross_encoder import get_reranker
from config import settings
from logger import get_logger

logger = get_logger(__name__)

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
        if settings.CROSS_ENCODER_ENABLED:
            try:
                self.cross_encoder = get_reranker(settings.RERANKER_MODEL_NAME)
                logger.info("Cross-encoder reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Cross-encoder not available: {e}")

        self._groq_semaphore = asyncio.Semaphore(_GROQ_CONCURRENCY)
        self._initialize_bm25_from_vectorstore()

    def _initialize_bm25_from_vectorstore(self):
        """Initialize BM25 index from vector store collection."""
        logger.info("Initializing BM25 index from vector store collection...")
        try:
            all_data = self.vector_store.collection.get()

            if not all_data:
                logger.info("No data returned from vector store.")
                return

            documents_list = all_data.get('documents')
            if not documents_list:
                logger.info("No documents found in vector store.")
                return

            metadatas_list = all_data.get('metadatas')

            documents = []
            if metadatas_list and len(metadatas_list) == len(documents_list):
                for doc_text, metadata in zip(documents_list, metadatas_list):
                    if metadata is None:
                        metadata = {}
                    documents.append(Document(page_content=doc_text, metadata=metadata))
            else:
                logger.warning("No metadata available, creating document objects without metadata.")
                for doc_text in documents_list:
                    documents.append(Document(page_content=doc_text, metadata={}))

            if documents:
                self.bm25.add_documents(documents)
                logger.info(f"Synced {len(documents)} documents to BM25 index.")
            else:
                logger.warning("No documents to add to BM25.")

        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"BM25 initialization failed: {e}") from e

    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================

    async def add_documents(self, documents, use_contextual=True, use_llm_context=True):
        """Add documents with optional contextualization."""
        if not use_contextual:
            return await self._add_documents_simple(documents)
        return await self._add_documents_with_context(documents, use_llm_context)

    async def _add_documents_simple(self, documents):
        """Add documents without contextualization."""
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
        """Add documents with contextualization."""
        doc_groups = self._group_by_source(documents)
        contextualized_docs = []

        for source, chunks in doc_groups.items():
            logger.info(f"Processing {len(chunks)} chunks from source '{source}'")
            doc_context = await self._get_document_context(chunks, use_llm)

            for batch_start in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[batch_start:batch_start + _BATCH_SIZE]
                batch_num = batch_start // _BATCH_SIZE + 1
                total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE
                logger.info(f"Contextualizing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

                tasks = [
                    self._contextualize_chunk_throttled(chunk, doc_context, use_llm)
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                contextualized_docs.extend(batch_results)

                if use_llm and batch_start + _BATCH_SIZE < len(chunks):
                    logger.info(f"Rate limit pause ({_BATCH_DELAY_SECONDS}s)...")
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
        """Group documents by source filename."""
        groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in groups:
                groups[source] = []
            groups[source].append(doc)
        return groups

    async def _get_document_context(self, chunks, use_llm):
        """Generate document-level context snippet."""
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
            logger.error(f"Context generation failed: {e}")
            return f"Document: {chunks[0].metadata.get('source', 'unknown')}"

    async def _contextualize_chunk_throttled(self, chunk, doc_context, use_llm):
        """Contextualize chunk with rate limiting."""
        async with self._groq_semaphore:
            result = await self._contextualize_chunk(chunk, doc_context, use_llm)
            if use_llm:
                await asyncio.sleep(0.5)
            return result

    async def _contextualize_chunk(self, chunk, doc_context, use_llm):
        """Add context header to chunk."""
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
                logger.error(f"Chunk contextualization failed: {e}")
                contextual_content = f"{doc_context}\n\n{chunk.page_content}"

        new_metadata = chunk.metadata.copy()
        new_metadata["original_content"] = chunk.page_content
        return Document(page_content=contextual_content, metadata=new_metadata)

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    async def retrieve(self, query, top_k=5, use_hybrid=True, use_reranking=True, alpha=0.5, score_threshold=0.0):
        """Main retrieval with optional hybrid search and cross-encoder reranking."""
        logger.info(f"Retrieval initiated: query='{query}', top_k={top_k}, hybrid={use_hybrid}, rerank={use_reranking}")
        fetch_k = top_k * 3 if use_reranking else top_k

        if use_hybrid:
            results = self._hybrid_search(query, top_k=fetch_k, alpha=alpha)
        else:
            results = self._vector_search(query, top_k=fetch_k)

        if not results:
            logger.info("No candidate documents returned from search indexes.")
            return []

        score_key = "combined_score" if use_hybrid else "similarity_score"

        filtered_results = [r for r in results if r.get(score_key, 0) >= score_threshold]
        logger.info(f"Score threshold {score_threshold} filtered out {len(results) - len(filtered_results)} candidates.")
        results = filtered_results

        if use_reranking and len(results) > top_k and self.cross_encoder:
            logger.info(f"Reranking {len(results)} candidates using CrossEncoder...")
            results = self.cross_encoder.rerank(
                query=query,
                documents=results,
                top_k=top_k,
                score_key="content"
            )
            top_score = results[0].get('final_score', results[0].get('combined_score', 0))
            logger.info(f"Cross-encoder reranking complete. Top score: {top_score:.4f}")

        final_results = results[:top_k]
        logger.info(f"Retrieval complete. Returning {len(final_results)} document chunks.")
        return final_results

    def _vector_search(self, query, top_k):
        """Pure vector search."""
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
        """Hybrid vector + BM25 search."""
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
        """Get system statistics."""
        try:
            return {
                "total_documents": self.vector_store.collection.count(),
                "collection_name": self.vector_store.collection_name,
                "bm25_documents": self.bm25.count(),
                "indices_synced": self.bm25.count() == self.vector_store.collection.count(),
                "cross_encoder_available": self.cross_encoder is not None
            }
        except Exception as e:
            logger.error(f"Error retrieving stats: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "bm25_documents": 0,
                "indices_synced": False,
                "cross_encoder_available": False
            }

    def clear(self):
        """Clear all documents."""
        self.vector_store.clear_collection()
        self.bm25.clear()
        logger.info("Cleared all documents from vector store and BM25 index.")