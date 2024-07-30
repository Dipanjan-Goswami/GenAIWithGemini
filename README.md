# GenAIWithGemini

This is Generative AI application which provides an insight to LLM Applications and use cases. In this Small Application I have shown capabilities of LLM with Chatbot using RAG(Retrieval Augmented Generation) and Memory. The Application finally integrates wtih langsmith Evolutor for Enabling monitoring and Tracing.

# Key Capabilties
This application for different python classes to acheive different capabilities

* Integrates with Google Gemini-1.5-Flash for Decoding and uses Text-embedding model
* Uses langchain and langsmith python library not to keep any vendor locking
* Demonstrate different types of prompts to showcase prompt engineering
* Utilized ConversationBufferMemory and ConversationalSummaryMemory to memorize ans summarize chat history
* Streamlit sessions to maintain session oriented chat with users
* utilizes chromadb and faiss - 2 open source vector databases to develop a RAG like system
* Finally used all the above concept into develop a chatbot.

# Pre-Requisite

To get started you need few things in place -
1. Python3.x installed in your local system
2. Google Gemini API key

# How to Get Started

* Create a new project or pull this project from Git. in case of new project you need to copy these files.
* Upgrade PIP and install virtualenv
* activate a virtual environment with following command 
  * > Mac - source venv/bin/activate
  * > Linux - source venv/bin/activate
  * > Windows - venv\Scripts\activate
* To get started you need to download the following python libraries using PIP.
  * langchain
  * langchain_core
  * langchain_community
  * langchain_google_genai
  * langchain_chroma
  * pypdf
  * chromadb
* Then go to config.json file and update the GoogleAI API Key
* Now you are good to test the files

# Referenced Files

* prompt-gemini.py -> Gives you an insight of prompt
* LC-memory-gemini.py -> Use of memory in a chat based systems
* streamlit-session.py -> Session management
* create-chroma-db.py -> Persist data into chromadb
* create-faiss-index.py -> Persist data into Faiss index vector stor
* chroma-db-retrieval.py -> Data retrieval from ChromaDB and push the external data to LLM
* faiss-retrieval.py -> Data retrieval from FAISS index and push the external data to LLM
* chatbot.py -> Final Chatbot Implementation

# How to start the chatbot

* To start the chatbot application, you need to run create-chroma-db.py file first.
* Once chromadb is created
* Execute the command which starts chromadb into your local and refers to chromadb folder
  > chroma run --host localhost --port 8000 --path ./chromadb
* Now start the streamlit chatbot by running the command
  > streamlit run chatbot.py
