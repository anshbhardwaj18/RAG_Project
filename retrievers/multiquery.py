from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning and deep learning"),
    Document(page_content="Gradient descent minimize the loss function."),
    Document(page_content="Gradient descent ia an optimization that minimizes the loss fucntion(or cost function)"),
    Document(page_content="Neural network use gradient descent for training."),
    Document(page_content="Support Vector Machines are Supervised learning algorithm")
]

embedding = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embedding)

retrievers = vectorstore.as_retriever()

llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retrievers = MultiQueryRetriever.from_llm(
    retriever= retrievers,
    llm = llm
)

query = "What is gradient descent?"

docs = multi_query_retrievers.invoke(query)

print("\nRetrived Documents:\n")

for doc in docs:
    print(doc.page_content)