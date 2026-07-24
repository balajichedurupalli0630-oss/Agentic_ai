from typing import List , Dict , Any 
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None 
        self._load_model()

    def _load_model(self):
        try :
            print(f"Loading CrossEncoder : {self.model_name }")
            self.model = CrossEncoder(self.model_name)
            print("Cross Encoder Loaded Successfully !!")
        except Exception as e :
            print(f"ERROR in Loading CrossEncoder : {self.model_name} as : {e}")
            raise 

    def rerank(self, 
               query : str ,
               documents : List[Dict[str , Any ]],
               top_k : int = 5 ,
               score_key : str = "content"

               ) -> List[Dict[str , Any ]]:
        

        if not documents:
            return []
        if not self.model :
            raise ValueError("Cross Encoder model not Loaded ...")
        pairs = [[query , doc.get(score_key," ")]for doc in documents]

        print(f"Reranking {len(documents)} documents with CrossEncoder ...")
        scores = self.model.predict(pairs)


        for doc , score in zip(documents , scores ):
            doc["cross_encoder_score"] = float(score) 

        reranked = sorted(documents , key=lambda x : x["cross_encoder_score"] , reverse=True )

        for i , doc in enumerate(reranked[:top_k]):
            doc["rerank_position"] = i + 1
            doc["final_score"] = doc["cross_encoder_score"]
        
        print(f"Reranked to top {top_k} results ")
        return reranked[:top_k]
    

_reranker = None 

def get_reranker(model_name : str  = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    global _reranker
    if _reranker is None or _reranker.model_name != model_name:
        _reranker = CrossEncoderReranker(model_name )
    return _reranker




