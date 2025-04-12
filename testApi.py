from fastapi.testclient import TestClient
from api import app  

client = TestClient(app)

def test_question_query():
    payload = {
        "query": "What is the capital of France?",
        "sourceDocs": ["Paris is the capital of France.", "France is a country in Europe."]
    }

    response = client.post("/question-query", json=payload)

    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "number_of_docs" in data
    assert data["number_of_docs"] == 2
    assert data["you are"] == "smelly"

    print("Test passed. Response:")
    print(data)

# Run this manually if not using pytest
if __name__ == "__main__":
    test_question_query()
