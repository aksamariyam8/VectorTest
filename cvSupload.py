
import os
import pinecone
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Initialize Pinecone using the new class-based approach
pc = pinecone.Pinecone(api_key='9b755b92-61ac-493d-b088-c35f36f7f2f3', environment='us-west1-gcp')

# Define index parameters
index_name = 'my-pdf-index'  # Your desired index name
index = pc.Index(index_name)  # Create an instance of the index

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Process PDFs in the specified folder
folder_path =  os.path.join(os.getcwd(), 'cvs')

for filename in os.listdir(folder_path):
     if filename.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(pdf_path)  # Extract text from PDF
        embedding = model.encode(text).tolist()  # Generate embedding from text

        # Prepare metadata (customize as needed)
        metadata = {
            'filename': filename,
            'length': len(text),  # Example: length of the text
            'uploaded_at': str(os.path.getmtime(pdf_path)),
            'text': text
        }

        # Upsert the vector into the index with metadata
        index.upsert(vectors=[(filename, embedding, metadata)])  # Use filename as the ID
        print(f"Uploaded '{filename}' to the index with metadata.")

print("All PDFs have been uploaded to the index.")



