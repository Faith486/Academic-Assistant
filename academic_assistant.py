import streamlit as st
import random
import time

from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory

import tempfile

load_dotenv()

# openai_key = os.getenv('OPENAI_KEY')
openai_key = st.secrets["OPENAI_KEY"]

st.write("Hey there! I am your study buddy! I'm here to help you prepare well for your exams!")

# Accept user input
uploaded_file = st.file_uploader("", type='pdf')

# Initialize chat history
if 'messages' not in st.session_state:
  st.session_state.messages = [{'role':'assistant', 'content':'What topic would you like to study? (You can upload a study note too)'}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])


# initiate memory to track conversation history
memory_without_rag = ConversationBufferMemory(memory_key='history', return_messages=True)
memory_with_rag = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# LLM model
llm = ChatOpenAI(model='gpt-4o-mini',
                  api_key=openai_key,
                  temperature=0,
                  verbose=0)

def update_assistant():
  if uploaded_file is not None:
    for message in st.session_state.messages:
      if message['role'] == 'assistant':
        memory_with_rag.chat_memory.add_ai_message(message['content'])
      elif message['role'] == 'user':
        memory_with_rag.chat_memory.add_user_message(message['content'])
  else:
    for message in st.session_state.messages:
      if message['role'] == 'assistant':
        memory_without_rag.chat_memory.add_ai_message(message['content'])
      elif message['role'] == 'user':
        memory_without_rag.chat_memory.add_user_message(message['content'])


if prompt := st.chat_input('Ask a question /Enter an answer'):
  # Add user message to chat history
  st.session_state.messages.append({'role': 'user', 'content': prompt})
  update_assistant()

  PROMPT_TEMPLATE = """
      You are a helpful AI assistant. Continue the conversation naturally.

      Conversation history:
      {history}

      User: {input}
    """
    
  prompt_template = PromptTemplate(
      input_variables=['input', 'history'],
      template=PROMPT_TEMPLATE
      )

  rag_chain = ConversationChain(
      llm=llm,
      memory=memory_without_rag,
      verbose=False,
    )

  if uploaded_file is not None:
    # Save file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
      tmp_file.write(uploaded_file.read())
      tmp_path = tmp_file.name
    
    chroma_db = tempfile.NamedTemporaryFile(dir='chroma_db', delete=False)

    # Use PyPDFloader
    file_loader = PyPDFLoader(tmp_path)
    docs = file_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    data = text_splitter.split_documents(docs)

    embedding_function = OpenAIEmbeddings(
      api_key=openai_key,
      model='text-embedding-3-small'
    )

    vecstore = Chroma.from_documents(
      data,
      embedding=embedding_function,
      persist_directory='chroma_db'
    )

    vecretriever = vecstore.as_retriever(
      search_type='similarity',
      search_kwargs={'k': 2}
    )

    docs = vecretriever.invoke("Give me information about Tommy's kitchen products?")
    
    PROMPT_TEMPLATE = """
      You are an helpful and friendly academic assistant who speaks English only. 
      You are to ask assessment questions randomly (one question at a time).
      Access whether the given response is an answer to the previously asked question or a random response (Don't give the user information about what you think about this)
      If you conclude that the user's response is more likely to be an random response more than an answer to a question, respond naturally (Don't give the user information about what you think about this).
      If you conclude that the user's response is more likely to be an answer more than a random response, provide an assessment score for the answer using the format X/10, where X is the individual's assessed score based on the answer provided and thereafter suggest an accurate answer using both the document and extensive domain knowledge from other related sources, using the following pieces of information

      `
        Context:
        {context}

        chat_history:
        {chat_history}

        Question:
        {question}
      `
      After each response, ask the next question on a completely new line using the format
      
      `
        Question: 
      `
    """
    
    prompt_template = PromptTemplate(
      input_variables=['context', 'chat_history', 'question'],
      template=PROMPT_TEMPLATE
      )

    rag_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vecretriever,
      memory=memory_with_rag,
      verbose=False,
      combine_docs_chain_kwargs={'prompt': prompt_template}
    )

  # Display user message in chat message container
  with st.chat_message('user'):
    st.markdown(prompt)

  # Display assistant response  in chat message container
  with st.chat_message('assistant'):
    message_placeholder = st.empty()
    full_response = ""
    
    assistant_response = rag_chain.run(prompt)

    # Simulate stream of response with milliseconds delay
    for chunk in assistant_response.split():
      full_response += chunk + " "
      time.sleep(0.05)
      # Add a blinking cursor to simulate typing
      message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
  # Add assistant response to chat history
  st.session_state.messages.append({'role': 'assistant', 'content': full_response})
  update_assistant()