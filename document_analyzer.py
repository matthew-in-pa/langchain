import streamlit as st
import io
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

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

def get_response(question: str, context: str, llm: init_chat_model):
    prompt = prompt_template.invoke({"context": context, "question": question})
    for response in st.session_state.llm.stream(prompt):
        yield response

def set_sessionstate_variables(document_type: str,embedding_model_name: str, embed_query_yes_no: str, lang_model: str):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    if "document_type" not in st.session_state:
        st.session_state.document_type = document_type

    if st.session_state.document_type != document_type:
        st.session_state.document_type = document_type

    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = embedding_model_name

    if st.session_state.embedding_model_name != embedding_model_name:
        st.session_state.embedding_model_name = embedding_model_name

    if "should_embed_query" not in st.session_state:
        st.session_state.should_embed_query = True if embed_query_yes_no == "Yes" else False

    should_embed_query = True if embed_query_yes_no == "Yes" else False
    if st.session_state.should_embed_query != should_embed_query:
        st.session_state.should_embed_query = should_embed_query

    if "lang_model" not in st.session_state:
        st.session_state.lang_model = lang_model

    if "llm" not in st.session_state: 
        st.session_state.llm =  init_chat_model(model=st.session_state.lang_model,model_provider="ollama", temperature=0.5)

    if st.session_state.lang_model != lang_model:
        st.session_state.lang_model = lang_model
        st.session_state.llm =  init_chat_model(model=st.session_state.lang_model,model_provider="ollama", temperature=0.5)

if os.path.exists("./datastore/ParisNotes.pdf"):
    os.remove("./datastore/ParisNotes.pdf")


document_types = ['PDF']

embedding_model_names = ['nomic-embed-text:latest','bge-m3:latest','jina/jina-embeddings-v2-base-en:latest']

models = ['mistral:latest','qwen3:latest','llama3.1:latest','phi4:latest','deepseek-r1:14b']

st.title("Document Reader & Analyzer")

st.sidebar.markdown("Search Options")

document_type = st.sidebar.selectbox("Choose Document Type",document_types)

embedding_model_name = st.sidebar.selectbox("Choose Embedding Model", embedding_model_names)

embed_query_yes_no = st.sidebar.radio("Embed Query",["Yes","No"])

lang_model = st.sidebar.selectbox("Choose Language Model",models)


system_template = """
You are a knowledgeable assistant trained to analyze and extract information from a context provided to you. Your task is to read the content carefully and answer questions based solely on the information present within the content. You cannot use any external knowledge, databases, or resources.
{context}
Please ensure to:

1. Focus on Content: Base all your answers strictly on the information contained in the PDF. Do not reference outside knowledge or suggest further reading.
2. Clarity in Responses: Provide clear, concise answers to the questions, ensuring that they are directly supported by the PDF's content.
3. Contextual Understanding: If a question refers to specific sections of the document, identify and quote those sections as necessary to support your answer.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),("user","{question}")]
)

set_sessionstate_variables(document_type= document_type
                           ,embedding_model_name=embedding_model_name
                           ,embed_query_yes_no=embed_query_yes_no
                           ,lang_model=lang_model)

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

doc_to_analyze = st.file_uploader("Choose a Document")
if doc_to_analyze is not None:
    pdf_bytes = doc_to_analyze.getvalue()
    pdf_path = f"./datastore/{doc_to_analyze.name}"
    with open(pdf_path,"wb") as file:
        file.write(pdf_bytes)
  
    vector_db = get_vectordb_fromPDF(pdf_path,st.session_state.embedding_model_name)
    st.session_state.vectordb = vector_db

if question := st.chat_input("How can I help you with this document"):
    st.chat_message("user").markdown(question)   

    results = ""
    if st.session_state.should_embed_query == False:
        results = st.session_state.vectordb.similarity_search(question)
    else:
        embedding = OllamaEmbeddings(model=st.session_state.embedding_model_name)
        embedded_query = embedding.embed_query(question)
        results = st.session_state.vectordb.similarity_search_by_vector(embedded_query)
    
     
    docs_content = "\n\n".join(result.page_content for result in results)

    response = st.chat_message("assistant").write_stream(get_response(question,docs_content, st.session_state.llm))
  
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})


