from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from LoadProperties import LoadProperties

from langchain_google_genai import GoogleGenerativeAIEmbeddings

## Step 1: Loading Data into corpus

pdf_loader = PyPDFDirectoryLoader("./pdf-docs" )
loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

#Step 2 - setup Google Generative AI Embedding

properties = LoadProperties()

embeddings = GoogleGenerativeAIEmbeddings(model=properties.getEmbeddingModel(), google_api_key=properties.getAPIkey(), task_type="retrieval_document")

#Step 3 - since GoogleGenerativeAIEmbeddings accepts only 96 documents in one run , we will input documents in batches.

# Set the batch size
batch_size = 96

# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

db = Chroma(embedding_function=embeddings , persist_directory="./chromadb")
retriever = db.as_retriever()

## Step 4: Save the document with start and end index

# Iterate over batches
for batch_num in range(num_batches):
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extract documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    # Your code to process each document goes here
    retriever.add_documents(batch_documents)
    print(start_index, end_index)

