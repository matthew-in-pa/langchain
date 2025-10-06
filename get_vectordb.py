from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
import os
def get_vectordb_fromPDF(doc_to_analyze, embedding_model_name: str) -> InMemoryVectorStore:
    loader = PyPDFLoader(doc_to_analyze)
    docs = loader.load()
    chunk_size = os.getenv("CHUNK_SIZE")
    chunk_overlap = os.getenv("CHUNK_OVERLAP")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap,add_start_index=True)
    splits = text_splitter.split_documents(docs)
    embedding_func = OllamaEmbeddings(model=embedding_model_name)
    vector_db = InMemoryVectorStore(embedding_func)
    vector_db.add_documents(documents=splits)
    return vector_db
