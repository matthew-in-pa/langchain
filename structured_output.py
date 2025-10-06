import streamlit as st
import io
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from get_vectordb import get_vectordb_fromPDF
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field

load_dotenv()

class VehicleRegistrationDetail(BaseModel):
    """Information about Vehicle Registration"""

    plate: str = Field(description="licence plate number of the vehicle")
    title: str = Field(description="TITLE number of the Vehicle")
    vin: str = Field(description="VIN number of the Vehicle")
    yearmake: str = Field(description="YR/MAKE of the vehicle")
    type: str = Field(description="Type of the vehicle")
    wid: str = Field(description="WID number of the vehicle")

def set_sessionstate_variables(lang_model: str):
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    if "lang_model" not in st.session_state:
        st.session_state.lang_model = lang_model

    if "llm" not in st.session_state: 
        st.session_state.llm =  init_chat_model(model=st.session_state.lang_model,model_provider="ollama", temperature=0.5)

    if st.session_state.lang_model != lang_model:
        st.session_state.lang_model = lang_model
        st.session_state.llm =  init_chat_model(model=st.session_state.lang_model,model_provider="ollama", temperature=0.5)


models = ['mistral:latest','qwen3:latest','llama3.1:latest','phi4:latest','deepseek-r1:14b']

st.title("PA Vehicle Registration Reader")

lang_model = st.sidebar.selectbox("Choose Language Model",models)


system_template = """
You are an expert at extracting structured information from documents.Extract all relevant information according to the schema provided.
"""

user_template ="""
Extract structured data from the following document:\n\n{text}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),("user",user_template)]
)


if os.path.exists("./datastore/Audi Registration_2025.pdf"):
    os.remove("./datastore/Audi Registration_2025.pdf")

set_sessionstate_variables(lang_model=lang_model)

doc_to_analyze = st.file_uploader("Choose a Document")
if doc_to_analyze is not None:
    pdf_bytes = doc_to_analyze.getvalue()
    pdf_path = f"./datastore/{doc_to_analyze.name}"
    with open(pdf_path,"wb") as file:
        file.write(pdf_bytes)
  
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunk_size = os.getenv("CHUNK_SIZE")
    chunk_overlap = os.getenv("CHUNK_OVERLAP")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap,add_start_index=True)
    splits = text_splitter.split_documents(docs)

    search_content = "\n\n".join(split.page_content for split in splits)

    prompt = prompt_template.invoke({"text": search_content})
    structured_llm = st.session_state.llm.with_structured_output(VehicleRegistrationDetail)

    vehicle_detail = structured_llm.invoke(prompt)     

    st.json(vehicle_detail)
    #st.markdown(f"{vehicle_detail}")                        
    
    




