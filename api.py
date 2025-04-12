#run with
#python -m uvicorn api:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define the expected input JSON format
class InputData(BaseModel):
    query: str

@app.post("/question-query")
async def custom_json_response(data: InputData):
    # You can build any custom response based on the input
    responseStr = "Hello, Keiran! You're old."
    # sourceDocs = len(data.sourceDocs)


    
    custom_response = {
        "response": responseStr,
        # "number_of_docs": sourceDocs,
        "you are": "smelly"
    }

    return custom_response
