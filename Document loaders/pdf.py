from langchain_classic.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token Based Splitting
data = PyPDFLoader("Document loaders/GRU.pdf")
docs = data.load()
splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)
chunks = splitter.split_documents(docs)
print(chunks[0].page_content)

# Semantic/Meaning Based Text Splitting
