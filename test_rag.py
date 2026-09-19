import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model used to create the knowledge base
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

# Get the existing collection
collection = client.get_collection("wellness_knowledge")

# Test student condition
query = "The student has high stress and very little sleep."

# Convert query into an embedding
query_embedding = embedding_model.encode(query).tolist()

# Retrieve the most relevant wellness information
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("=" * 60)
print("RAG RETRIEVAL TEST")
print("=" * 60)

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\nRelevant Knowledge {i}:")
    print(document)

print("\n" + "=" * 60)
print("RAG retrieval test completed successfully!")
print("=" * 60)