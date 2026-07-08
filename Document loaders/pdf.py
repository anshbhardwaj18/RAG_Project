from langchain_classic.document_loaders import PyPDFLoader

data = PyPDFLoader("Document loaders/GRU.pdf")

docs = data.load()
print(docs)
print(len(docs))