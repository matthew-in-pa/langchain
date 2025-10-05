from langchain.chat_models import init_chat_model

llm = init_chat_model(model="mistral:latest",model_provider="ollama", temperature=0.5)

response = llm.invoke("What is the capital of India?")

print(response.content)