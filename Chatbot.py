import streamlit as st
from langchain_groq import ChatGroq
import openai
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="ALL CHATBOT"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant.PLease response to user's every query"),
        ("user","Question:{question}")
    ]
)


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def generate_response(question,llm,temp):
    if "gpt" in llm:
        llm = ChatOpenAI(model=llm,temperature=temp)
    elif "gemini" in llm:
        llm = ChatGoogleGenerativeAI(model=llm, temperature=temp)
    elif "llama" in llm or "gemma" in llm:
        llm = ChatGroq(model=llm, groq_api_key=groq_api_key, temperature=temp)
    else:
        return "Selected model is not supported yet."
    
    output=StrOutputParser()
    chain=prompt|llm|output
    answer=chain.invoke({'question':question})
    return answer


st.title("Enchanced Chatbot")
st.sidebar.title("Settings")

llm=st.sidebar.selectbox("Select model",["gpt-4o","gpt-4-turbo","gemini-2.0-flash","gemini-1.5-pro","llama-3.1-8b-instant","gemma2-9b-it"])
temperature=st.sidebar.slider("Temperature",min_value=0.00,max_value=1.00,value=0.70)

input=st.text_input("You:")

if input:
    response=generate_response(input,llm,temperature)
    st.write(response)
