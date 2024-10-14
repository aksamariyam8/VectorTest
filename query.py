
import os
import pinecone
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
# Initialize Pinecone using the new class-based approach
pc = pinecone.Pinecone(api_key='9b755b92-61ac-493d-b088-c35f36f7f2f3', environment='us-west1-gcp')

# Define index parameters
index_name = 'my-pdf-index'  # Your desired index name

index = pc.Index(index_name)

# Generate a valid embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
test_text = "aqsa"
valid_embedding = model.encode(test_text).tolist()

# Normalize the embedding
valid_embedding = np.array(valid_embedding)
norm = np.linalg.norm(valid_embedding)
if norm > 0:
    valid_embedding = (valid_embedding / norm).tolist()

# Query the index
try:
    results = index.query(
        vector=valid_embedding, 
        top_k=1,
        include_metadata=True,
        include_values=True
    )

    # Process and print the results
    if results and results['matches']:
        for match in results['matches']:
            # Extracting and printing relevant information
            print(match)
            text = match.get('metadata', {}).get('text', 'No text available')
            print(f"Match ID: {match['id']}, Score: {match['score']}, Text: {text}")
    else:
        print("No matches found.")
        
except pinecone.exceptions.PineconeException as e:
    print("Pinecone error:", e)
except Exception as e:
    print("Error querying index:", e)