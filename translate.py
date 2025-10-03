from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# MODEL = os.getenv("MODEL")


system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),("user","{text}")]
)

languages = ['German','Italian','French','Spanish']

models = ['mistral:latest','qwen3:latest','llama3.1:latest','phi4:latest','deepseek-r1:14b']

st.title("Translation Service")

st.sidebar.markdown("## SIDEBAR HEADER ##")

language = st.sidebar.selectbox("Choose Language", languages)
model = st.sidebar.selectbox("Choose Language Model", models)


llm = ChatOllama(model=model,temperature=0.5)

def translate_message(message):
    prompt = prompt_template.invoke({"language": language, "text": message})
    for response in llm.stream(prompt):
        yield response.content



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("What do you want to know?"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    response = st.chat_message("assistant").write_stream(translate_message(question))
    st.session_state.messages.append({"role": "assistant", "content": response})




