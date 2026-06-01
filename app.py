from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("movies.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Generate Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Store
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

# User Query
query = input("Enter your query: ")

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