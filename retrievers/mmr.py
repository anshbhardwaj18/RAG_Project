from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning and deep learning"),
    Document(page_content="Gradient descent minimize the loss function."),
    Document(page_content="Gradient descent ia an optimization that minimizes the loss fucntion(or cost function)"),
    Document(page_content="Neural network use gradient descent for training."),
    Document(page_content="Support Vector Machines are Supervised learning algorithm")
]

embedding = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embedding)

similarity_retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

print("\n===== Similarity Search Result =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent")

for doc in similarity_docs:
    print(doc.page_content)

mmr_retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

print("\n===== MMR Result =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent")

for doc in mmr_docs:
    print(doc.page_content)