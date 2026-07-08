from langchain_community.document_loaders import TextLoader

data = TextLoader("Document loaders/notes.txt")

docs = data.load()

print(docs[0])