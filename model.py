#This is basically a pdf loader
from google.generativeai import configure, GenerativeModel

from PyPDF2 import PdfReader
from chonkie import SemanticChunker
from modelManager import PineconeModelManager 

#Rip rip all the plaintext from the pdfile
def extract_text_from_pdf(pdf_file_path):
    # Extract text from the entire PDF document
    pdf_text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

#The CHUNK
def chunk_text(text):

    chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-8M",  # Default model
    threshold=0.6,                               # Similarity threshold (0-1) or (1-100) or "auto"
    chunk_size=512,                              # Maximum tokens per chunk
    min_sentences=1                              # Initial sentences per chunk
    ) 
    
    chunks = chunker.chunk(text)
    
    chunkies=[]
    for chunk in chunks:
        chunkies.append(chunk.text)
    return chunkies


def parseDocs(file_name: str ="sample_data/requests-readthedocs-io-en-latest.pdf",):
    print("Extracting Text .....")
    pdf_text = extract_text_from_pdf(file_name)

    print("Chunking .....")

    text_chunks = chunk_text(pdf_text)
    return text_chunks
    

def query(query: str, doc_model :PineconeModelManager, gemini_model : GenerativeModel):
    retrieved_docs=doc_model.query(query)
    print(query)
    prompt1 = f"Use ONLY the context provided to answer the query. Context: {retrieved_docs}\n\nQuestion: {query}\n\nAnswer:"

    print(" We retrieved thse docs\n")
    print(retrieved_docs)

    print(f"\n\nQuery was: ", query)

    response1 = gemini_model.generate_content(prompt1)

    return response1


# retrieved_docs = retrieve_relevant_docs(query)
# print("DOCS",retrieved_docs)


# def generate_answer(query, context, gemini_model):
#     prompt1 = f"Use ONLY the context provided to answer the query. Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     response1 = gemini_model.generate_content(prompt1)
#     prompt2 = f"answer this question {query}"
#     response2=  gemini_model.generate_content(prompt2)
#     return response1.text, response2.text

# answer1, answer2 = generate_answer(query, retrieved_docs)

# print("Query:", query)
# print("\nRetrieved Context:", retrieved_docs[0])
# print("\nRAG Answer:", answer1)

# print("\nNormal Answer:", answer2)

