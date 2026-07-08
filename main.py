from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.document_loaders import PyPDFLoader

load_dotenv()

# data = TextLoader("Document loaders/notes.txt")  # This is when you load your txt file and read that 
data2 = PyPDFLoader("Document loaders/GRU.pdf")  # This is when you load your pdf file and read that
# docs = data.load()
docs2 = data2.load()
template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarizes the text"),
    ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-small-2506")
prompt = template.format_messages(data = docs2[0].page_content)
result = model.invoke(prompt)
print(result.content)