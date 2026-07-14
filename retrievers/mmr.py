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