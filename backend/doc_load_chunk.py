from logging import WARNING
from  langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader
)
from typing import Any, List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import exists
def split_documents(documents : List[Any],chunk_size :int = 800, chunk_overlap : int = 160):
    """ Split documents into smaller chunks for better Rag performace """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap,
        length_function = len,
        separators=["\n\n", "\n\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks ")

    if split_docs :
        print(f"\n Example Chunk :")
        print(f"content : {split_docs[0].page_content[:200]}")
        print(f"Metadata : {split_docs[0].metadata}")
    return split_docs

def load_documents(data_path : str = "data") -> List[Any]:
            data_path = Path(data_path).resolve()
            documents = []

            loaders = [
        ("*.pdf", PyMuPDFLoader),
        ("*.txt", TextLoader),
        ("*.csv", CSVLoader),
    ]
            if not data_path.exists():
                 print(f"[WARNING] Diractory {data_path} does not exist . Creating it ..")
                 data_path.mkdir(parents=True, exist_ok = True)
                 return documents

            for pattern, LoaderClass in loaders:
                for file in data_path.rglob(pattern):
                    try:
                        loader = LoaderClass(str(file))
                        loaded = loader.load()
                        documents.extend(loaded)
                        print(f"Loaded: {file.name}")
                    except Exception as e:
                        print(f"[ERROR] Could not load {file}: {e}")

            print(f"Total loaded documents: {len(documents)}")
            return documents
def load_single_document(data_path : str ) -> List[Any]:
     """ 
     Load single document 
     
     """
     data_path = Path(data_path)

     if not data_path.exists():
          raise FileNotFoundError(f"File not Found : {data_path}")
     extension = data_path.suffix.lower()

     loader_map = {
          '.pdf' : PyMuPDFLoader,
          '.txt' : TextLoader,
          '.csv' : CSVLoader,

     }
     Loader_class = loader_map.get(extension)
     if not Loader_class:
          raise ValueError(f"Unsupported file type : {extension}")
     try: 
          loader = Loader_class(str(data_path))
          documents = loader.load()
          print(f"Loaded : {data_path.name} ({len(documents)}) documents ")
          return documents 
     except Exception as e :
          print(f"[ERROR] Could not Load : {data_path} : {e}")
          raise 

