from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import chromadb
from langchain_chroma import Chroma
from langchain.memory.buffer import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from LoadProperties import LoadProperties


## Step 1: Setup environment by loading properties and GenAI Models
properties = LoadProperties()

generation_config = {
    "temperature": 1,
    "top_p": 0.30,
    "top_k": 10,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

llm = GoogleGenerativeAI(model=properties.getModelName(), google_api_key=properties.getAPIkey(), generation_config =  generation_config)

embeddings = GoogleGenerativeAIEmbeddings(model=properties.getEmbeddingModel(), google_api_key=properties.getAPIkey(), task_type="retrieval_document")

## Step 2: Setup chromadb

client = chromadb.HttpClient(host="127.0.0.1", port=8000)
db = Chroma(client=client, embedding_function=embeddings)

## Step 3: Setup Retriever

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
docs = retv.invoke('Tell us About Dipanjan')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

## Step 4: Using RetrievalQA fetch details from chromadb

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv,return_source_documents=True)

response = chain.invoke("Tell us about Dipanjan's certifications")

print(response)


## Step 5: Using ConversationBufferMemory with RetrievalQA, Fetch details from chromadb and save chat into memory

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key="llm_output", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retv, memory=memory, output_key="llm_output", return_source_documents=True)

response = qa.invoke({"question": "Tell us about Dipanjan's Skill set"})
print(memory.chat_memory.messages)

print(response)

response = qa.invoke({"question": "Tell us about Dipanjan's Hobbies"})
print(memory.chat_memory.messages)

print(response)