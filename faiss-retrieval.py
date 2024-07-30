from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
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

## Step 2: Setup FAISS db

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

## Step 3: Setup Retriever

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
docs = retv.invoke('Tell us which module is most relevant to LLMs and Generative AI')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

## Step 4: Using RetrievalQA fetch details from faiss vector store

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv,return_source_documents=True)

response = chain.invoke("Tell us about Dipanjan's Hobbies")

print(response)


