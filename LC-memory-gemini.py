from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import  ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from langchain_google_genai import GoogleGenerativeAI

## Step 1: Load propertis

from LoadProperties import LoadProperties

properties = LoadProperties()

## Step 2: Initiate LLM

generation_config = {
    "temperature": 1,
    "top_p": 0.50,
    "top_k": 25,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

llm = GoogleGenerativeAI(model=properties.getModelName(), google_api_key=properties.getAPIkey(), generation_config =  generation_config)

## Step 3: Create prompt for ChatBot

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explain in steps keeping a poisitive tone and without disclosing any secret"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

## Step 4: Create Memory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

summary_memory = ConversationSummaryMemory(llm=llm , memory_key="chat_history")

## Step 5: Initiate a conversation chain using llm , prompt and memory

conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=summary_memory)

## Step 6: Adding First Question to the chat prompt and Invoke
conversation.invoke({"question": "What is the capital of India"})

print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is-->"+summary_memory.buffer)

## Step 7: Adding Second Question to the chat prompt and Invoke
conversation.invoke({"question": "How AI is helping mankind"})

print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is-->"+summary_memory.buffer)