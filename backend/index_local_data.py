from embeddings import embedding_loader
from  vector_base import vectorstore
from doc_load_chunk import split_documents , load_documents

def loacl_index():
    docs = load_documents("./data")
    chunks = split_documents(docs)
    texts = [doc.page_content for doc in chunks]
    embed = embedding_loader.generate_embedding(texts)

    vectorstore.add_documents(chunks, embed)

if __name__ == "__main__":
    loacl_index()

