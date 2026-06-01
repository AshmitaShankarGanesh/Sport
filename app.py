from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

PDF_PATH = "sports.pdf"
VECTOR_DB_PATH = "faiss_db"

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Check if FAISS DB already exists
if os.path.exists(VECTOR_DB_PATH):

    print("Loading existing FAISS database...")

    vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("Creating new FAISS database...")

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create FAISS Vector Store
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save locally
    vector_db.save_local(VECTOR_DB_PATH)

    print("FAISS database saved locally.")

# User Query
query = input("\nEnter your query: ")

# Similarity Search
results = vector_db.similarity_search(
    query,
    k=3
)

# Display Results
for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Page Number: {doc.metadata['page'] + 1}")

    print(doc.page_content)

    print("-" * 80)