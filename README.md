🧠 Multimodal AI Chatbot with MemoryWelcome to the Multimodal AI Chatbot, a versatile and intelligent assistant built with Streamlit and LangChain. This application leverages the power of various large language models (LLMs) to provide a seamless conversational experience across different modes, including general chat, document analysis (PDFs), and image interpretation.✨ Key FeaturesMulti-Modal Interaction: Switch effortlessly between different modes:Normal Mode: Engage in general conversation with session memory.PDF Mode: Upload one or more PDFs and ask questions about their content. The chatbot uses a RAG (Retrieval-Augmented Generation) pipeline to provide context-aware answers.Image Mode: Upload an image and ask questions about what it contains.Flexible LLM Support: Easily select from a wide range of powerful models from providers like Google (Gemini) and Groq (Llama, Gemma).Persistent Session History: Conversations are saved per session ID, allowing you to pick up where you left off.Real-time Streaming: Responses are streamed token-by-token for a dynamic and interactive user experience.Modern Chat UI: A clean and intuitive interface built with Streamlit's latest chat components.🛠️ Tech StackThis project is built on a modern stack of AI and web technologies:Application Framework: StreamlitCore AI/LLM Framework: LangChainLLM Providers:langchain-google-genai (for Gemini models)langchain-groq (for high-speed Llama, Gemma, etc.)Embeddings & Vector Store:sentence-transformers (for generating text embeddings)langchain-chroma & chromadb (for efficient in-memory vector storage)Document Loading: pypdf (for PDF processing)Environment Management: python-dotenv🚀 Getting StartedFollow these steps to set up and run the project locally.1. PrerequisitesPython 3.8 or higherGit2. Clone the Repositorygit clone <your-repository-url>
cd <your-repository-name>
3. Set Up a Virtual EnvironmentIt's highly recommended to use a virtual environment to manage dependencies.# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install DependenciesInstall all the required packages using the requirements.txt file.pip install -r requirements.txt
5. Configure Environment VariablesCreate a file named .env in the root of your project directory. You will need to get API keys from their respective providers.# Get from Google AI Studio
GOOGLE_API_KEY="your-google-api-key"

# Get from GroqCloud Console
GROQ_API_KEY="your-groq-api-key"

# Optional: For LangSmith Tracing
LANGCHAIN_API_KEY="your-langchain-api-key"
🏃 How to RunOnce the setup is complete, you can run the Streamlit application with a single command:streamlit run your_main_script.py
Replace your_main_script.py with the actual name of your Python file. The application will open in a new tab in your web browser. Enjoy your intelligent chatbot!
