# 🧠 Multimodal AI Chatbot with Memory & Download
LINK-https://enchanced-chatbot-kv8dftq8zwcanpywhgfrzd.streamlit.app/

Welcome to the **Multimodal AI Chatbot**, a versatile and intelligent assistant built with **Streamlit** and **LangChain**.  
This application leverages the power of various large language models (LLMs) to provide a seamless conversational experience across different modes, including **general chat**, **document analysis (PDFs)**, and **image interpretation**.

---

## ✨ Key Features

- **Multi-Modal Interaction**: Switch effortlessly between different modes:
  - **Normal Mode**: Engage in general conversation with session memory.
  - **PDF Mode**: Upload one or more PDFs and ask questions about their content. The chatbot uses a **RAG (Retrieval-Augmented Generation)** pipeline to provide context-aware answers.
  - **Image Mode**: Upload an image and ask questions about what it contains using powerful multimodal models.

- **Flexible LLM Support**: Easily select from a wide range of powerful models from providers like **Google (Gemini)** and **Groq (Llama, Gemma, etc.)**.

- **Persistent Session History**: Conversations are saved per session ID, allowing you to pick up where you left off.

- **Real-time Streaming**: Responses are streamed token-by-token for a dynamic and interactive user experience.

- **Download Chat History**: Easily download the current conversation as a `.txt` file with a single click.

- **Modern Chat UI**: A clean and intuitive interface built with Streamlit's latest chat components.

---

## 🛠️ Tech Stack

This project is built on a modern stack of AI and web technologies:

- **Application Framework**: [Streamlit](https://streamlit.io)
- **Core AI/LLM Framework**: [LangChain](https://www.langchain.com)
- **LLM Providers**:
  - `langchain-google-genai` (for Gemini models)
  - `langchain-groq` (for high-speed Llama, Gemma, etc.)
- **Embeddings & Vector Store**:
  - `sentence-transformers` (for generating text embeddings)
  - `langchain-chroma` & `chromadb` (for efficient in-memory vector storage)
- **Document Loading**: `pypdf` (for PDF processing)
- **Environment Management**: `python-dotenv`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.8+**
- Git

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-name>
