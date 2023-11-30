import streamlit as sl
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
import google.generativeai as palm
from langchain.llms.google_palm import GooglePalm
from langchain.embeddings.google_palm import GooglePalmEmbeddings

from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import nltk
nltk.download('punkt')
from langchain.text_splitter import NLTKTextSplitter


from htmlCode import styles, human_class, robot_class, fade_in_css
import time
import os

# API key for the google palm 2 model and embeddings
os.environ["GOOGLE_API_KEY"] = "AIzaSyCeVFdUtFDGjce1ODc3FrL2GtW5JOGesAo"

def get_text_and_split(pdfs):
    raw_text = ""
    for pdf_doc in pdfs:
        pdf_reader = PdfReader(pdf_doc)
        for page_num in pdf_reader.pages:
            text = page_num.extract_text()
            raw_text += text       
    return raw_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(chunks):
    embeddings = GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def conversation(vectorstore):
    language_model = GooglePalm(temperature=0.5)

    memory = ConversationBufferMemory(
        memory_key = 'chat_history',    
        return_messages=True,
    )

    conversation = ConversationalRetrievalChain.from_llm(   
        llm = language_model,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    return conversation



def answer_question(asked_question):
    # Get the answer from the conversation chain
    print(asked_question)
    answer = sl.session_state.chat_conversation({'question': asked_question})

    print(answer)


    sl.session_state.chat_history = answer['chat_history']

    # Displaying the conversation.
    for i, message in enumerate(sl.session_state.chat_history):
        print(f"Message {i}: {message.content}\n")
        if i % 2 == 0:
            sl.write(human_class.replace(
                "{{human message}}", message.content), unsafe_allow_html=True)
        else:
            sl.write(robot_class.replace(
                "{{robot message}}", message.content), unsafe_allow_html=True)
            

def main():
    load_dotenv()
    sl.set_page_config(page_title="PDF chat", page_icon="")

    sl.write(styles, unsafe_allow_html=True)

    if "chat_conversation" not in sl.session_state:
        sl.session_state.chat_conversation = None

    if "chat-history" not in sl.session_state:
        sl.session_state.chat_history = None

    sl.header("PDF chat")
    question = sl.text_input("Type your question regarding your uploaded pdf's...")

    sl.markdown(fade_in_css, unsafe_allow_html=True)
    sl.write(human_class.replace("{{human message}}", "Yo, robot! Please answer my questions on my pdfs!"), unsafe_allow_html=True)
    time.sleep(2)
    sl.write(robot_class.replace("{{robot message}}", "Hello, human. I will glady do so. Just upload your pdfs!"), unsafe_allow_html=True)

    if question:
        answer_question(question)

    with sl.sidebar:
        sl.subheader('Your uploaded :red[documents]')
        pdfs = sl.file_uploader("Upload multiple PDF's", accept_multiple_files=True)
        col1, col2, col3 = sl.columns(3)
        with col2:
            if sl.button("Process PDF's"):
                with sl.spinner("Processing PDF's..."): 
                    # Getting the text and splitting into chunks from pdfs
                    text = get_text_and_split(pdfs)
                    chunks = get_text_chunks(text)
                    vectorstore = vector_store(chunks)
                   
                    sl.session_state.chat_conversation = conversation(vectorstore)
    

if __name__ == '__main__':
    main()