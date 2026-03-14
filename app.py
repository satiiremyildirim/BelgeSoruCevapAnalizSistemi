# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import asyncio
import tempfile
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time
from langchain_community.document_loaders import PyPDFLoader

# Const
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
GENERATION_MODEL_NAME  = "gemini-2.5-pro"
EMBEDDING_MODEL = "gemini-embedding-001"
RETRIEVER_K = 4

load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def configure_page():
    st.set_page_config(
        page_title="Document QA",
    )

    st.title("Document QA Analysis System")




def handle_new_document_button():
    if st.sidebar.button("Document", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]


def sidebar():
    
   
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
    )

    st.sidebar.subheader("Questions and Answers")

    col1, col2 = st.sidebar.columns(2)

    handle_new_document_button()

    st.sidebar.divider()

    message_count = len(st.session_state.messages) - 1 
    document_processed = (
        "retriever" in st.session_state
        and st.session_state.get("retriever") is not None
    )

    col1, col2 = st.sidebar.columns(2)
   
    uploaded_file = st.file_uploader(
        type=["pdf", "txt"],
        help="Upload a PDF or text file to chat with",
    )

    return selected_model, uploaded_file, st.session_state.get("api_key")


def handle_document_processing(uploaded_file=""):
    if st.button("Doc", type="primary"):
        user_api_key = st.session_state.get("api_key", "")
        if not user_api_key:
            st.error("API KEY")
            return
        else:
                try:
                    
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:  # txt file
                        loader = TextLoader(tmp_file_path)

                    documents = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = splitter.split_documents(documents)
                    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
                    )

                    st.session_state["retriever"] = retriever
                    st.session_state["document_name"] = uploaded_file.name

                    os.unlink(tmp_file_path)

                    st.success("Document analyzed.")
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed_for_cache: str | None):
    return ChatGoogleGenerativeAI(model=model_name)


def display_chat_messages():
    for message in st.session_state.messages[1:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

init_session_state()
configure_page()
selected_model, uploaded_file, user_api_key = handle_sidebar()
handle_document_processing(uploaded_file)
if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
    chat_model = get_chat_model(selected_model, user_api_key)


display_chat_messages()



def handle_user_input(chat_model, input_disabled: bool = False):
    if prompt := st.chat_input(
        "Ask", disabled=input_disabled
    ):
        st.session_state.messages.append(HumanMessage(content=prompt))

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
        )

        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if st.chat_message("assistant"):
                try:
                    retrieved_docs = retriever.invoke(prompt)                    
                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                    )
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt_template | chat_model | parser

                    message_placeholder = st.empty()
                    full_response = ""

                    for chunk in main_chain.stream(prompt):
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(
                                full_response + ""
                            ) 

                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                   
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    

handle_user_input(chat_model, input_disabled=(chat_model is None))
