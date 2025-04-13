import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import fitz
from google.generativeai import configure, GenerativeModel
import json
from PyPDF2 import PdfReader


#insert API keys, configure DB


def load_config(file_path="config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config

# Read and display credentials
config = load_config()

gemini_api_key=config['gemini']
pinecone_api_key =config['pinecone']


pinecone_env = "us-east-1"  # Example: 'us-east-1'

configure(api_key=gemini_api_key)
pc = Pinecone(
        api_key=pinecone_api_key,
  )



index_name = "workshop" #name for your database

existing_indexes = [index["name"] for index in pc.list_indexes()]

# create only if it doesn't exist already
if index_name not in existing_indexes:
  pc.create_index(
      name=index_name,
      dimension= 384, # Replace with your model dimensions (this embedding model has 384 dimensions)
      metric= "cosine", # Replace with your model metric
      spec=ServerlessSpec(
          cloud="aws",
          region="us-east-1"
      )
  )
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    embeddings = model.encode(text)  # The model expects a list of texts
    return embeddings


def extract_text_from_pdf(pdf_file_path):
    # Extract text from the entire PDF document
    pdf_text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def chunk_text(text, chunk_size=500):
    # Split text into smaller chunks based on the chunk_size (e.g., 500 characters)
    # This is a simple approach, there are ways to do better (having overlap, etc)
    chunks = []
    print("Total Chunks to be processed: ", len(text)//chunk_size)
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def ingest_pdf(pdf_file_path, doc_id):
    print("Starting Text Extraction...")
    pdf_text = extract_text_from_pdf(pdf_file_path)
    print("Starting Chunking ...")
    text_chunks = chunk_text(pdf_text, chunk_size = 500 ) #mention chunk size
    print("Starting the Embedding process ...")
     # Generate embeddings for all text chunks at once
    embeddings = embed_text(text_chunks)
    print(embeddings.shape)
    print("Data prepared to upsert")
    # Prepare upsert payload
    upsert_data = [
        (f"{doc_id}_{i}", embedding.tolist(), {"text": chunk})
        for i, (embedding, chunk) in enumerate(zip(embeddings, text_chunks))
    ]
    # Upsert all at once
    index.upsert(upsert_data)
    print("data upserted :)")


# Ingest an uploaded PDF document:
# ingest_pdf("sample_data/CMSC226-32842 Syllabus Spring25.pdf", "doc1") #doc1 is a sample identifier, you can make it anything

def retrieve_relevant_docs(query, top_k=5):
    query_embedding = embed_text([query]) #the function expects a list of vectors
    query_embedding = query_embedding.tolist()  # Convert ndarray to list

    # Query Pinecone for the top_k most relevant chunks
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Return the relevant text chunks
    return [match["metadata"]["text"].replace("\n","") for match in results["matches"]] 


gemini_model = GenerativeModel(model_name="gemini-2.0-flash")


query = "Which rooms are does the professor hold office hours?" #user query
retrieved_docs = retrieve_relevant_docs(query)
print("DOCSS",retrieved_docs)


def generate_answer(query, context):
    prompt1 = f"Use ONLY the context provided to answer the query. Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response1 = gemini_model.generate_content(prompt1)
    prompt2 = f"answer this question {query}"
    response2=  gemini_model.generate_content(prompt2)
    return response1.text, response2.text

answer1, answer2 = generate_answer(query, retrieved_docs)

print("Query:", query)
print("\nRetrieved Context:", retrieved_docs[0])
print("\nRAG Answer:", answer1)

print("\nNormal Answer:", answer2)

