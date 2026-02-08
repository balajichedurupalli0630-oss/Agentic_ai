import re
from typing import List, Any, Dict
from embeddings import Embeddings
from vector_base import VectorStore

class RAG:
    def __init__(self, vectorstore: VectorStore, embedding_loader: Embeddings, llm):
        self.vector_store = vectorstore
        self.embedding_loader = embedding_loader
        self.llm = llm
    
    async def retrieve(self, query: str, top_k: int = 15, score_threshold: float = 0.3, use_expansion: bool = False) -> List[Dict[str, Any]]:
        """Async document retrieval with DEFENSIVE CHECKS"""
        print(f"Retrieving documents for query: {query}")
        
        if use_expansion:
            expanded_queries = await self.aexpand_query(query)
        else:
            expanded_queries = [query]
            
        retrieved_docs = []
        
        for que in expanded_queries:
            print(f"Processing query: {que}")
            
            query_embedding = self.embedding_loader.generate_embedding([que])[0]
            
            try:
                results = self.vector_store.collections.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                # DEFENSIVE: Ensure results is dict
                if not isinstance(results, dict):
                    print(f"[ERROR] Expected dict, got {type(results)}")
                    continue
                
                if 'documents' not in results or not results['documents']:
                    print("[WARNING] No documents in results")
                    continue
                
                # Safely extract lists
                documents_list = results['documents'][0] if results['documents'] else []
                metadatas_list = results['metadatas'][0] if results['metadatas'] else []
                distances_list = results['distances'][0] if results['distances'] else []
                ids_list = results['ids'][0] if results['ids'] else []
                
                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids_list, documents_list, metadatas_list, distances_list)):
                    
                    # DEFENSIVE: Ensure document is string
                    if isinstance(document, list):
                        document = str(document)
                    
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata if isinstance(metadata, dict) else {},
                            'similarity_score': round(similarity_score, 4),
                            'distance': round(distance, 4),
                            'rank': i + 1,
                            'query': que
                        })
                    
            except Exception as e:
                print(f"[ERROR] in retrieval: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Deduplicate
        seen_ids = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc['id'] not in seen_ids:
                seen_ids.add(doc['id'])
                unique_docs.append(doc)
        
        print(f"Retrieved {len(unique_docs)} unique documents")
        return unique_docs
    
    async def aexpand_query(self, query: str) -> List[str]:
        """Async query expansion"""
        prompt = f"""Given this search query: {query}
        
Generate 3 alternative search queries:
1. A more detailed version
2. A question format
3. A keyword-based version

Return only search queries, one per line."""
        
        expanded_queries = [query]
        
        try:
            res = await self.llm.ainvoke(prompt)
            content = res.content if hasattr(res, 'content') else str(res)
            
            lines = content.split("\n")
            for line in lines:
                clean_line = re.sub(r'^[\d\.\-\)]*\s*', '', line.strip())
                if clean_line and len(clean_line) > 3:
                    expanded_queries.append(clean_line)
                    
        except Exception as e:
            print(f"Error in query expansion: {e}")
            
        return expanded_queries[:4]