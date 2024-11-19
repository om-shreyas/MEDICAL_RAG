from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings

def load_doc(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def get_embedding_function():
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    return embeddings

# print(get_embedding_function())

# for doc in documents:
#     print(doc.page_content)
