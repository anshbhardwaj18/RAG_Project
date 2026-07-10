from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# data = TextLoader("Document loaders/notes.txt")  # This is when you load your txt file and read that 
data = PyPDFLoader("Document loaders/deeplearning.pdf")  # This is when you load your pdf file and read that
# docs = data.load()
docs2 = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs2)


template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarizes the text"),
    ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-small-2506")
prompt = template.format_messages(data = docs2)
result = model.invoke(prompt)
print(result.content)