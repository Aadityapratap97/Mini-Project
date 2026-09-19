import chromadb
from sentence_transformers import SentenceTransformer

# Load local embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# Create / connect to local ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Reset existing collection
try:
    client.delete_collection("wellness_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="wellness_knowledge"
)

# Read knowledge base
with open("wellness_knowledge.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Split knowledge using topic headings
raw_sections = content.split("\n\n")

documents = []

current_topic = None
current_content = []

for section in raw_sections:
    section = section.strip()

    if not section:
        continue

    # Skip main title
    if section == "ACADEMIC WELLNESS KNOWLEDGE BASE":
        continue

    # Detect topic headings
    if section.isupper():
        if current_topic and current_content:
            documents.append(
                f"{current_topic}\n\n{' '.join(current_content)}"
            )

        current_topic = section
        current_content = []

    else:
        current_content.append(section)

# Add final section
if current_topic and current_content:
    documents.append(
        f"{current_topic}\n\n{' '.join(current_content)}"
    )

# Generate embeddings
embeddings = embedding_model.encode(documents).tolist()

# Store in ChromaDB
collection.add(
    ids=[f"wellness_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings
)

print("=" * 50)
print("Knowledge base created successfully!")
print(f"Documents stored: {len(documents)}")
print("ChromaDB database: chroma_db")
print("=" * 50)