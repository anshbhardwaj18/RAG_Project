# load pdf
# split into chunks
# Create the embeddings
# store into chroma db
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

data = PyPDFLoader("Document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma_db"
)

print("Total Chunks :", len(chunks))
print("Stored Documents :", vectorstore._collection.count())
print(os.getcwd())
print(Path("chroma_db").resolve())