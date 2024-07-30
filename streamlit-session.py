from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain_google_genai import GoogleGenerativeAI
from LoadProperties import LoadProperties

## Step 1: Load properties and Set LLM object

properties = LoadProperties()

generation_config = {
    "temperature": 1,
    "top_p": 0.50,
    "top_k": 25,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

llm = GoogleGenerativeAI(model=properties.getModelName(), google_api_key=properties.getAPIkey(), generation_config =  generation_config)

## Step 2: Create Streamlit Chat history Object

history = StreamlitChatMessageHistory(key="chat_messages")

## Step 3: here we create a memory object

memory = ConversationBufferMemory(chat_memory=history)

## Step 4: Creation of Chat prompt

## Case 5: Getting response using SystemMessage and HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("ou are an AI chatbot having a conversation with a human"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
#Step 6 - here we use streamlit to print all messages in the memory, create text imput, run chain and
#the question and response is automatically put in the StreamlitChatMessageHistory

import streamlit as st

st.title('🦜🔗 Welcome to the ChatBot')
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message("human").write(x)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    response = llm_chain.invoke(x)
    st.chat_message("ai").write(response["text"])
