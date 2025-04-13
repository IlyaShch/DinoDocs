from abc import abstractmethod
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec



class PineconeModelManager:
    def __init__(self, api_key: str, environment: str, index_name: str):
        print("Beggining Init Sequence")
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        # pinecone.init(api_key=self.api_key, environment=self.environment)
        self.pc = Pinecone( api_key=self.api_key)

        self.index = self._ensure_index()

        self.index = self.pc.Index(index_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("INit COmplete")

    def _ensure_index(self):
        """Ensure the index is initialized or created."""
        
        existing_indexes = [index["name"] for index in self.pc.list_indexes()]

        # create only if it doesn't exist already we really only need to do this when starting a model.
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension= 384, # Replace with your model dimensions (this embedding model has 384 dimensions)
                metric= "cosine", # Replace with your model metric
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )



    def upsert(self,embeddings, items: List[Any], namespace: Optional[str] = None, doc_id: str="doc0"):
        """Upsert multiple items into the index."""
        upsert_data = [
            (f"{doc_id}_{i}", embedding.tolist(), {"text": chunk})
            for i, (embedding, chunk) in enumerate(zip(embeddings, items))
        ]
        # Upsert all at once
        self.index.upsert(upsert_data,namespace)
        print("data upserted :)")

    def query(self, query, top_k: int = 5, namespace: Optional[str] = None) -> Dict:
        """Query the index with a vector."""
        # print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSHIIIITT",type(query))

        try:
            query_embedding = self.embed_text([query])[0].tolist()

            # Query Pinecone for the top_k most relevant chunks
            results = self.index.query(vector=query_embedding, top_k=top_k,include_metadata=True)

            #Testing what we have
            print(f"Got {len(results['matches'])} matches for query: {query}")

            # Return the relevant text chunks
            return [match["metadata"]["text"].replace("\n","") for match in results["matches"]]
        except Exception as e:
            print("Exception during query")
            return [] 

    def delete(self, ids: List[str], namespace: Optional[str] = None):
        """Delete vectors by ID."""
        self.index.delete(ids=ids, namespace=namespace)

    def embed_text(self, text):
        self.embeddings = self.model.encode(text)  # The model expects a list of texts
        return self.embeddings
      
    def describe_index_stats(self) -> Dict:
        """Get index statistics."""
        return self.index.describe_index_stats()
    
    