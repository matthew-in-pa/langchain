from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

file_path ="./data/ParisNotes.pdf"

loader = PyPDFLoader(file_path=file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

all_splits = text_splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

# for split in all_splits:
#     print(split.page_content)
#     print()
#     print(embedding.embed_query(split.page_content))
#     print()

vector_store = InMemoryVectorStore(embedding)

ids = vector_store.add_documents(documents=all_splits)

#Search Without Embedding the query
# results = vector_store.similarity_search(
#     "How to get to Paris from CDG airport"
# )

#search without emdedding but also return the similarity score
# results = vector_store.similarity_search_with_score(
#     "How to get to Paris from CDG airport"
# )

#search based on the similarity to the embedded query
#emdedded_query = embedding.embed_query("How to get to Paris from CDG Airport")
emdedded_query = embedding.embed_query("Neighborhoods")
results = vector_store.similarity_search_with_score_by_vector(emdedded_query)

doc, score = results[0]
print(doc)
print()
print(score)