from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")
result = model.invoke("Hello")
print(result.content)