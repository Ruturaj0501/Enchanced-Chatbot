__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import io
import base64
import asyncio

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "ALL CHATBOT"
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
groq_api_key = os.getenv('GROQ_API_KEY')

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

def get_llm_instance(llm_name, temp):
    if "gemini" in llm_name:
        return ChatGoogleGenerativeAI(model=llm_name, temperature=temp)
    elif "openai/gpt-oss" in llm_name:
        return ChatGroq(model_name=llm_name, groq_api_key=groq_api_key, temperature=temp)
    elif "llama" in llm_name or "gemma" in llm_name or "deepseek" in llm_name:
        return ChatGroq(model_name=llm_name, groq_api_key=groq_api_key, temperature=temp)
    else:
        st.error("Selected model is not supported yet.")
        return None

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.set_page_config(page_title="🧠 Enchance Chatbot")
st.title("🧠 Enchance Chatbot")
st.sidebar.title("Settings")

mode = st.sidebar.radio("Choose Mode", ("Normal", "PDF", "Image"))

if mode == "Image":
    st.sidebar.info("Image analysis requires a Gemini (multimodal) model.")
    model_choices = ["gemini-2.5-flash", "gemini-2.5-pro"]
else:
    model_choices = [
        "gemini-flash-latest",
        "gemini-2.5-pro",
        "deepseek-r1-distill-llama-70b",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma2-9b-it",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"
    ]

llm_selection = st.sidebar.selectbox("Select Model", model_choices)
temperature = st.sidebar.slider("Temperature", min_value=0.00, max_value=1.00, value=0.70)

session_id = st.text_input("Session ID", value="default_session")

retriever = None

if mode == "PDF":
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing PDFs and syncing with Pinecone..."):

            @st.cache_resource(show_spinner=False)
            def process_pdfs(uploaded_files):
                documents = []
                for file in uploaded_files:
                    temp_path = f"./{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_path)
                    loaded_docs = loader.load()
                    
                    total_text = "".join([doc.page_content for doc in loaded_docs])
                    if not total_text.strip():
                        st.warning(f"⚠️ '{file.name}' seems to be an image or scanned PDF. No text could be extracted!")
                    
                    documents.extend(loaded_docs)
                    os.remove(temp_path)

                if not documents:
                    return None

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                index_name = "enhanced-chatbot"
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                
                existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
                if index_name not in existing_indexes:
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    while not pc.describe_index(index_name).status["ready"]:
                        time.sleep(1)
                
                PineconeVectorStore.from_documents(
                    documents=splits, 
                    embedding=embeddings, 
                    index_name=index_name
                )
                
                time.sleep(3)
                
                vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                return vectorstore.as_retriever(search_kwargs={"k": 10})

            retriever = process_pdfs(uploaded_files)
            if retriever:
                st.success("PDFs processed and ready for querying!")

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image.", use_container_width=True)
        st.session_state.image_bytes = uploaded_image.getvalue()

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

history = get_session_history(session_id)

for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if user_question := st.chat_input("Ask your question here..."):
    with st.chat_message("human"):
        st.markdown(user_question)
    history.add_user_message(user_question)

    llm_instance = get_llm_instance(llm_selection, temperature)
    if llm_instance is None:
        st.stop()

    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""

        if mode == "PDF" and retriever:
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Given a chat history and the latest user question... formulate a standalone question..."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
            history_aware_retriever = create_history_aware_retriever(llm_instance, retriever, contextualize_q_prompt)
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an assistant for question-answering tasks... Use the following retrieved context to answer...\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm_instance, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            response_stream = rag_chain.stream(
                {"input": user_question, "chat_history": history.messages}
            )

            def extract_answer(stream):
                for chunk in stream:
                    if 'answer' in chunk:
                        yield chunk['answer']

            full_response = response_placeholder.write_stream(extract_answer(response_stream))

        elif mode == "Image" and "image_bytes" in st.session_state:
            message = HumanMessage(content=[
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode()}"}
            ])
            response_stream = llm_instance.stream([message])

            def extract_content(stream):
                for chunk in stream:
                    yield chunk.content

            full_response = response_placeholder.write_stream(extract_content(response_stream))

        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's questions."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            chain = prompt | llm_instance | StrOutputParser()

            response_stream = chain.stream(
                {"input": user_question, "chat_history": history.messages}
            )
            full_response = response_placeholder.write_stream(response_stream)

    if full_response:
        history.add_ai_message(full_response)
