import re
from pathlib import Path
from typing import List, Any
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def is_table_block(block: str) -> bool:
    """
    Detects if a markdown block is a table.
    A table block starts with '|' and has a separator line starting with '|'
    containing only '-', ':', '|', and whitespace.
    """
    block = block.strip()
    lines = block.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) < 2:
        return False
    
    # First line must start with '|'
    if not lines[0].startswith('|'):
        return False
        
    # Second line must start and end with '|' and contain only markdown table separators
    line2 = lines[1]
    if not line2.startswith('|') or not line2.endswith('|'):
        return False
        
    if not re.match(r'^[\s|:-]+$', line2):
        return False
        
    if '-' not in line2:
        return False
        
    return True

def load_and_chunk_pdf_with_tables(pdf_path: str) -> List[Document]:
    """
    Load PDF using pymupdf4llm.to_markdown() and chunk it.
    Tables are preserved as whole chunks with content_type='table'.
    Other paragraphs/sections are chunked normally with content_type='text'.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract markdown representation of the PDF
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    
    # Split the markdown into blocks by blank lines
    raw_blocks = re.split(r'\n\s*\n', md_text)
    
    # Text splitter for prose chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        length_function=len,
        separators=["\n\n", "\n\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    
    for block in raw_blocks:
        block_cleaned = block.strip()
        if not block_cleaned:
            continue
            
        if is_table_block(block_cleaned):
            # Keep table block as a single chunk
            chunks.append(Document(
                page_content=block_cleaned,
                metadata={
                    "source": str(pdf_path),
                    "content_type": "table"
                }
            ))
        else:
            # Prose block: split normally using character splitter
            sub_docs = text_splitter.create_documents(
                texts=[block_cleaned],
                metadatas=[{"source": str(pdf_path), "content_type": "text"}]
            )
            chunks.extend(sub_docs)
            
    return chunks
