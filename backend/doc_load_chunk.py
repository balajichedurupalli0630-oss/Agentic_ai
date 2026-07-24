from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from typing import Any, List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from doc_load_chunk_tables import load_and_chunk_pdf_with_tables


def split_documents(documents: List[Any], chunk_size: int = 800, chunk_overlap: int = 160):
    """Split documents into smaller chunks for better RAG performance.

    Documents already chunked by the table-aware PDF loader (metadata
    _pre_chunked=True) are passed through untouched, so table blocks
    never get re-split mid-row.
    """
    already_chunked = [d for d in documents if d.metadata.get("_pre_chunked")]
    to_split = [d for d in documents if not d.metadata.get("_pre_chunked")]

    split_docs = list(already_chunked)
    if to_split:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n\n\n", "\n", ". ", " ", ""],
        )
        new_splits = text_splitter.split_documents(to_split)
        for d in new_splits:
            if "content_type" not in d.metadata:
                d.metadata["content_type"] = "text"
        split_docs.extend(new_splits)

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks "
          f"({len(already_chunked)} pre-chunked table-aware, {len(split_docs) - len(already_chunked)} newly split)")
    if split_docs:
        print(f"Example chunk:\n{split_docs[0].page_content[:200]}")
    return split_docs


def load_documents(data_path: str = "data") -> List[Any]:
    data_path = Path(data_path).resolve()
    documents = []

    loaders = [
        ("*.pdf", PyMuPDFLoader),
        ("*.txt", TextLoader),
        ("*.csv", CSVLoader),
    ]

    if not data_path.exists():
        print(f"[WARNING] Directory {data_path} does not exist. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
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

    print(f"Total loaded: {len(documents)} documents")
    return documents


def load_single_document(data_path: str, original_filename: str = None) -> List[Any]:
    """Load a single document by path.

    PDFs are routed through the table-aware loader (returns pre-chunked
    Documents, with tables kept intact as single chunks). TXT/CSV keep the
    original loader path and get chunked normally downstream by
    split_documents().
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix == ".pdf":
        try:
            chunks = load_and_chunk_pdf_with_tables(str(data_path))
            num_table = sum(1 for c in chunks if c.metadata.get("content_type") == "table")
            num_text = sum(1 for c in chunks if c.metadata.get("content_type") == "text")
            display_name = original_filename if original_filename else data_path.name
            print(f"[TABLE-AWARE LOAD] {display_name}: {len(chunks)} chunks ({num_table} table, {num_text} text)")
            # Mark as pre-chunked so split_documents() downstream can skip re-splitting
            for c in chunks:
                c.metadata["_pre_chunked"] = True
                if original_filename:
                    c.metadata["source"] = original_filename
            return chunks
        except Exception as e:
            print(f"[ERROR] Table-aware load failed for {data_path}, falling back to PyMuPDFLoader: {e}")
            loader = PyMuPDFLoader(str(data_path))
            documents = loader.load()
            if original_filename:
                for c in documents:
                    c.metadata["source"] = original_filename
            return documents

    loader_map = {
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }
    LoaderClass = loader_map.get(suffix)
    if not LoaderClass:
        raise ValueError(f"Unsupported file type: {suffix}")

    try:
        loader = LoaderClass(str(data_path))
        documents = loader.load()
        if original_filename:
            for c in documents:
                c.metadata["source"] = original_filename
        display_name = original_filename if original_filename else data_path.name
        print(f"Loaded: {display_name} ({len(documents)} pages/rows)")
        return documents
    except Exception as e:
        print(f"[ERROR] Could not load {data_path}: {e}")
        raise