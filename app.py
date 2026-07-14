import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat With Your PDF")

# --------------------------
# Session State
# --------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------
# Upload PDF
# --------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# --------------------------
# Create Database
# --------------------------

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully")

    if st.button("Create Knowledge Base"):

        with st.spinner("Creating Embeddings..."):

            # Delete previous database

            if os.path.exists("chroma_db"):
                try:
                    shutil.rmtree("chroma_db")
                except:
                    pass

            # Load PDF

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            # Split

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            chunks = splitter.split_documents(docs)

            # Embedding

            embedding_model = MistralAIEmbeddings()

            # Vector DB

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db",
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5,
                },
            )

            llm = ChatMistralAI(
                model="mistral-small-2506"
            )

            st.session_state.retriever = retriever
            st.session_state.llm = llm

        st.success(f"Knowledge Base Created ✅ ({len(chunks)} Chunks)")

# --------------------------
# Prompt
# --------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not present in the context,
say:

I could not find the answer in the document.
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# --------------------------
# Chat
# --------------------------

if st.session_state.retriever:

    # Old Messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input

    question = st.chat_input(
        "Ask anything about the PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = st.session_state.retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = st.session_state.llm.invoke(
            final_prompt
        )

        with st.chat_message("assistant"):
            st.markdown(response.content)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)