from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()



embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function= embedding_model
)

print("Collection Count:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

# PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             """You are a helpful AI assistant.

Use Only the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""     ),
        (
        "human",
        """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG System Created")

print("Press 0  to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")

# ***************************** THIS IS USE WHEN YOU LOAD TEXT FROM PDF AND TEXT FILE --------------------------->>>>>>>>>>>>>>>>>>>>>
# from langchain_community.document_loaders import TextLoader
# from langchain_classic.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # data = TextLoader("Document loaders/notes.txt")  # This is when you load your txt file and read that 
# data = PyPDFLoader("Document loaders/deeplearning.pdf")  # This is when you load your pdf file and read that
# # docs = data.load()
# docs2 = data.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunks = splitter.split_documents(docs2)


# template = ChatPromptTemplate.from_messages(
#     [("system", "You are a AI that summarizes the text"),
#     ("human", "{data}")]
# )
# model = ChatMistralAI(model="mistral-small-2506")
# prompt = template.format_messages(data = docs2)
# result = model.invoke(prompt)
# print(result.content)

