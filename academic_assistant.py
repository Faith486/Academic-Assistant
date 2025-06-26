import streamlit as st
import random
import time

from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory

import tempfile

# Securely get OpenAI key
openai_key = st.secrets["OPENAI_KEY"]

st.write("Hey there! I am your study buddy! I'm here to help you prepare well for your exams!")

# Accept user input
uploaded_file = st.file_uploader("", type='pdf')

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'What topic would you like to study? (You can upload a study note too)'}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Memory
memory_without_rag = ConversationBufferMemory(memory_key='history', return_messages=True)
memory_with_rag = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# LLM setup
llm = ChatOpenAI(model='gpt-4o-mini', api_key=openai_key, temperature=0, verbose=0)

def update_assistant():
    mem = memory_with_rag if uploaded_file else memory_without_rag
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            mem.chat_memory.add_ai_message(message['content'])
        elif message['role'] == 'user':
            mem.chat_memory.add_user_message(message['content'])

if prompt := st.chat_input('Ask a question /Enter an answer'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    update_assistant()

    prompt_template = PromptTemplate(
        input_variables=['input', 'history'],
        template="""
        You are a helpful AI assistant. Continue the conversation naturally.

        Conversation history:
        {history}

        User: {input}
        """
    )

    rag_chain = ConversationChain(
        llm=llm,
        memory=memory_without_rag,
        verbose=False
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        file_loader = PyPDFLoader(tmp_path)
        docs = file_loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data = splitter.split_documents(docs)

        embedding_function = OpenAIEmbeddings(api_key=openai_key, model='text-embedding-3-small')
        vecstore = FAISS.from_documents(data, embedding_function)

        vecretriever = vecstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})

        rag_prompt = PromptTemplate(
            input_variables=['context', 'chat_history', 'question'],
            template="""
            You are a helpful academic assistant. Ask one assessment question at a time.
            If the user's response is not a proper answer, respond naturally.
            If it is a proper answer, score it out of 10, and provide a correct answer.

            Context:
            {context}

            Chat History:
            {chat_history}

            Question:
            {question}

            Next question must be asked on a new line using:
            Question:
            """
        )

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vecretriever,
            memory=memory_with_rag,
            verbose=False,
            combine_docs_chain_kwargs={'prompt': rag_prompt}
        )

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        placeholder = st.empty()
        response = ""
        assistant_reply = rag_chain.run(prompt)
        for word in assistant_reply.split():
            response += word + " "
            time.sleep(0.05)
            placeholder.markdown(response + "▌")
        placeholder.markdown(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
    update_assistant()