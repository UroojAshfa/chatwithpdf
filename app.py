import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv


load_dotenv()
#load GROQ and GoogleAIEmbeddings from .env file

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model PDF chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>

Questions: {input}
"""

)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./data") #Data Ingestion
        st.session_state.docs = st.session_state.loader.load() #Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


prompt1 = st.text_input("What would you like to know?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector store db is ready")


import time

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input' : prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------")

else:
    if not prompt1:
        st.write("Please enter a query.")
    if "vectors" not in st.session_state:
        st.write("Please create the vector store before querying.")