from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os


class RAGQueryModel:
    def __init__(self):
        # Llama3 model through Ollama
        self.model = ChatOllama(model="llama3:latest")
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # Prompt template for querying
        self.prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        # Internal components
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def _load_files(self, folder_path):
        """Loads and processes all supported files in a folder."""
        supported_loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
        }

        docs = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            ext = os.path.splitext(file_name)[-1].lower()
            loader_class = supported_loaders.get(ext)
            if loader_class:
                loader = loader_class(file_path)
                docs.extend(loader.load())
            else:
                print(f"Unsupported file format: {file_name}")

        return docs

    def ingest_folder(self, folder_path):
        """Ingests files from a folder and creates a vector store."""
        # Load documents
        docs = self._load_files(folder_path)
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        # Create a vector store
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # Create a retriever with similarity search
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 3,
                'score_threshold': 0.5
            }
        )
        # Build the RAG chain
        self.chain = ({
            "context": self.retriever,
            "question": RunnablePassthrough()
        }
        | self.prompt
        | self.model
        | StrOutputParser()
        )

    def ask(self, query: str):
        """Asks a question using the RAG model."""
        if not self.chain:
            return "Please ingest a folder with documents first."
        return self.chain.invoke(query)

    def clear(self):
        """Clears the vector database and retriever."""
        self.vector_store = None
        self.retriever = None
        self.chain = None


# Example Usage
if __name__ == "__main__":
    # Initialize the model
    rag_model = RAGQueryModel()
    
    # Ingest files from a folder
    folder_path = "data"  # Change to your folder path
    rag_model.ingest_folder(folder_path)
    
    # Query the RAG model
    question = "What is the reinforcement Learning?"
    answer = rag_model.ask(question)
    print("Answer:", answer)
