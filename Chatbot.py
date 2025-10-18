# Chatbot.py — updated to avoid pysqlite3 / chroma sqlite OperationalError on Streamlit Cloud

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
import logging
import io
import base64
import streamlit as st
from dotenv import load_dotenv

# LLM & LangChain imports
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# chromadb (we will create client explicitly)
import chromadb
from chromadb.utils import embedding_functions

# Setup dotenv & env
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "ALL CHATBOT"

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
groq_api_key = os.getenv('GROQ_API_KEY')

# Logging — helpful locally (Streamlit Cloud will still redact error details in the UI)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Cached resources --------------------
# Cache embeddings and the (optional) persistent client to avoid reinitialization
@st.cache_resource
def get_embeddings():
    # CPU-friendly model; change device if you have GPU
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

@st.cache_resource
def create_chroma_client(persist: bool = False, persist_dir: str = "/tmp/chroma_db"):
    """
    Return a chroma client:
      - If persist==False: returns an ephemeral in-memory Chromadb client (safe on Streamlit Cloud)
      - If persist==True: returns a PersistentClient using persist_dir (use only if directory is writable & no concurrent writers)
    """
    if persist:
        # ensure path exists
        os.makedirs(persist_dir, exist_ok=True)
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Created PersistentClient at {persist_dir}")
            return client
        except Exception as e:
            logger.exception("Failed to create PersistentClient; falling back to ephemeral client.")
            return chromadb.Client()
    else:
        # in-memory ephemeral client (recommended for Streamlit Cloud demos)
        logger.info("Using ephemeral chromadb.Client() (in-memory) — recommended for Streamlit Cloud.")
        return chromadb.Client()

# get resources
embeddings = get_embeddings()
# default: do NOT persist to filesystem on Streamlit Cloud (avoids sqlite issues)
chroma_client = create_chroma_client(persist=False)

# ------------------- LLM INSTANCE GENERATOR -------------------
def get_llm_instance(llm_name, temp):
    """Returns an instance of the selected LLM."""
    if "gemini" in llm_name:
        return ChatGoogleGenerativeAI(model=llm_name, temperature=temp, convert_system_message_to_human=True)
    elif "openai/gpt-oss" in llm_name:
        return ChatGroq(model_name=llm_name, groq_api_key=groq_api_key, temperature=temp)
    elif "llama" in llm_name or "gemma" in llm_name or "deepseek" in llm_name:
        return ChatGroq(model_name=llm_name, groq_api_key=groq_api_key, temperature=temp)
    else:
        st.error("Selected model is not supported yet.")
        return None

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="🧠 Enchance Chatbot")
st.title("🧠 Enchance Chatbot")
st.sidebar.title("Settings")

mode = st.sidebar.radio("Choose Mode", ("Normal", "PDF", "Image"))

if mode == "Image":
    st.sidebar.info("Image analysis requires a Gemini (multimodal) model.")
    model_choices = ["gemini-2.5-flash", "gemini-2.5-pro"]
else:
    model_choices = ["gemini-flash-latest","gemini-2.5-pro","deepseek-r1-distill-llama-70b","llama-3.1-8b-instant","llama-3.1-70b-versatile","gemma2-9b-it","openai/gpt-oss-20b","openai/gpt-oss-120b"]

llm_selection = st.sidebar.selectbox("Select Model", model_choices)
temperature = st.sidebar.slider("Temperature", min_value=0.00, max_value=1.00, value=0.70)

session_id = st.text_input("Session ID", value="default_session")

retriever = None

# ------------------- FILE HANDLING & MODE SPECIFIC UI -------------------
if mode == "PDF":
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing PDFs..."):

            @st.cache_resource(show_spinner=False)
            def process_pdfs(_uploaded_files):
                """
                - Load PDFs, split to chunks, build a Chroma vectorstore.
                - If Chromadb persistent storage fails (sqlite issues), fall back to in-memory client.
                """
                documents = []
                for file in _uploaded_files:
                    temp_path = f"./{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                    os.remove(temp_path)

                # Split text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)

                # Try to create a Chroma vectorstore using the chroma_client we created above.
                # If there is an OperationalError from sqlite/chromadb, attempt to recreate an ephemeral client.
                try:
                    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, client=chroma_client)
                    logger.info("Chroma vectorstore created using provided client.")
                    return vectorstore.as_retriever()
                except Exception as e:
                    # log details locally
                    logger.exception("Failed to create Chroma vectorstore with provided client. Falling back to ephemeral in-memory client.")
                    # If fallback: use ephemeral client
                    fallback_client = create_chroma_client(persist=False)
                    try:
                        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, client=fallback_client)
                        logger.info("Chroma vectorstore created using ephemeral client fallback.")
                        return vectorstore.as_retriever()
                    except Exception:
                        logger.exception("Fallback in-memory Chroma creation also failed.")
                        raise

            retriever = process_pdfs(uploaded_files)
            st.success("PDFs processed and ready!")

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image.", use_container_width=True)
        st.session_state.image_bytes = uploaded_image.getvalue()

# ------------------- CHAT HISTORY MANAGEMENT -------------------

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

history = get_session_history(session_id)

# show previous messages
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# ------------------- CHAT INPUT AND RESPONSE LOGIC -------------------

if user_question := st.chat_input("Ask your question here..."):
    # Display user message and add to history
    with st.chat_message("human"):
        st.markdown(user_question)
    history.add_user_message(user_question)

    llm_instance = get_llm_instance(llm_selection, temperature)
    if llm_instance is None:
        st.stop()

    # Assistant's response logic
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""

        if mode == "PDF" and retriever:
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", "Given a chat history and the latest user question... formulate a standalone question..."), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            history_aware_retriever = create_history_aware_retriever(llm_instance, retriever, contextualize_q_prompt)
            qa_prompt = ChatPromptTemplate.from_messages(
                [("system", "You are an assistant for question-answering tasks... Use the following retrieved context to answer...\n\n{context}"), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            question_answer_chain = create_stuff_documents_chain(llm_instance, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            response_stream = rag_chain.stream(
                {"input": user_question, "chat_history": history.messages}
            )

            def extract_answer(stream):
                for chunk in stream:
                    if isinstance(chunk, dict) and 'answer' in chunk:
                        yield chunk['answer']
                    # some chains may stream plain strings or objects — be permissive:
                    elif isinstance(chunk, str):
                        yield chunk

            full_response = response_placeholder.write_stream(extract_answer(response_stream))

        elif mode == "Image" and "image_bytes" in st.session_state:
            message = HumanMessage(content=[
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode()}"}
            ])
            response_stream = llm_instance.stream([message])

            def extract_content(stream):
                for chunk in stream:
                    # chunk may be message-like
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk:
                        yield chunk["content"]
                    else:
                        yield str(chunk)

            full_response = response_placeholder.write_stream(extract_content(response_stream))

        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's questions."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            chain = prompt | llm_instance | StrOutputParser()

            response_stream = chain.stream(
                {"input": user_question, "chat_history": history.messages}
            )
            # allow the stream to be raw chunks (strings or objects)
            def normalize_stream(stream):
                for chunk in stream:
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        yield chunk['content']
                    elif hasattr(chunk, "content"):
                        yield chunk.content
                    else:
                        # best-effort fallback
                        yield str(chunk)

            full_response = response_placeholder.write_stream(normalize_stream(response_stream))

    if full_response:
        history.add_ai_message(full_response)
