from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


#Text Based Splitting
splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap = 1
)

data = TextLoader("Document loaders/notes2.txt")

docs = data.load()
chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
    print()