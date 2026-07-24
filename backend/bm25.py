from typing import List, Dict, Any
from langchain_core.documents import Document
import re


class BM25:
    def __init__(self):
        self.bm25_index = None 
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return 
        self.documents.extend(documents)
        self._rebuild_index()
        print(f"Added {len(documents)} documents to BM25 of total ({len(self.documents)})")

    def _rebuild_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized)
        except ImportError:
            print("ERROR in Building BM25 index")

    def get_scores(self, query: str) -> Dict[int, float]:
        """
        Get BM25 scores for all documents (for hybrid search).
        
        Args:
            query: Search query string
            
        Returns:
            Dict mapping document content hash to BM25 score
        """
        if not self.bm25_index or not self.documents:
            return {}
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores from BM25Okapi
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Map document content hash to score
        score_dict = {}
        for doc, score in zip(self.documents, scores):
            doc_hash = hash(doc.page_content)
            score_dict[doc_hash] = float(score)
        
        return score_dict
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def clear(self) -> None:
        """Clear all documents"""
        self.documents = []
        self.bm25_index = None
        print("✓ BM25 index cleared")
    
    def count(self) -> int:
        """Return number of documents"""
        return len(self.documents)