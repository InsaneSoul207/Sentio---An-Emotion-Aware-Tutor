import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, pdf_path, api_key):
        os.environ["GOOGLE_API_KEY"] = api_key
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        
        self.vector_store = self.process_pdf(pdf_path)

    def process_pdf(self, path):
        loader = PyPDFLoader(path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(pages)

        return FAISS.from_documents(docs, self.embeddings)

    def search_context(self, query):
        results = self.vector_store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in results])