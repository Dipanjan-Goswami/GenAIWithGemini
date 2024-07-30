from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import chromadb
from langchain_chroma import Chroma
from langchain.memory.buffer import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from LoadProperties import LoadProperties


def create_chain():
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

    client = chromadb.HttpClient(host="127.0.0.1", port=8000)
    db = Chroma(client=client, embedding_function=embeddings)

    retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retv , output_key='answer', memory=memory,
                                               return_source_documents=True)
    return qa


def chat(user_message):
    # generate a prediction for a prompt
    bot_json = chain.invoke({"question": user_message})
    print(bot_json)
    return {"bot_response": bot_json}

chain = create_chain()

if __name__ == "__main__":
    import streamlit as st
    st.subheader("Portfolio Chatbot powered by Google Generative AI Service")
    col1 , col2 = st.columns([4,1])

    user_input = st.chat_input()
    with col1:
        col1.subheader("------Ask me a question about my career------")
        #col2.subheader("References")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if user_input:
            bot_response = chat(user_input)
            st.session_state.messages.append({"role" : "chatbot", "content" : bot_response})
            #st.write("OU Assistant Response: ", bot_response)
            for message in st.session_state.messages:
                st.chat_message("user")
                st.write("Question: ", message['content']['bot_response']['question'])
                st.chat_message("assistant")
                st.write("Answer: ", message['content']['bot_response']['answer'])
                #with col2:
                st.chat_message("assistant")
                # for doc in message['content']['bot_response']['source_documents']:
                #     st.write("Reference: ", doc.metadata['source'] + "  \n"+ "-page->"+str(doc.metadata['page']))
