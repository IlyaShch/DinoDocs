#run with
#python -m uvicorn api:app --reload
import json
from fastapi import FastAPI
from pydantic import BaseModel
from modelManager import PineconeModelManager 
from google.generativeai import configure, GenerativeModel
import model as m
##########################SETUP AREA ###############################
def load_config(file_path="config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config

config = load_config()
gemini_api_key=config['gemini']
pinecone_api_key =config['pinecone']

#Parsing Text
text_chunks=m.parseDocs()

#Setting up model
print("Starting Model Setup:")
myModel=PineconeModelManager(pinecone_api_key, "us-east-1", "workshop") 

print("Upserting Model!")
myModel.upsert(myModel.embed_text(text_chunks),text_chunks)



configure(api_key=gemini_api_key)
gemini_model = GenerativeModel(model_name="gemini-2.0-flash")

#Example Query
# query = "How do I POST a request?" #user query
# response=m.query(query,myModel,gemini_model)
# print(response.text)

app = FastAPI()

# Define the expected input JSON format
class InputData(BaseModel):
    query: str


##########################SEX AREA ###############################
@app.post("/question-query")
async def custom_json_response(data: InputData):
    # You can build any custom response based on the input
    responseStr = "Hello, Keiran! You're old."
    # sourceDocs = len(data.sourceDocs)

    
    response=m.query(query,myModel,gemini_model)
    
    custom_response = {
        "response": response,
        # "number_of_docs": sourceDocs,
        "you are": "smelly"
    }

    return custom_response


