from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from langchain_community.document_loaders import CSVLoader, PyPDFLoader,TextLoader
from langchain.chains import RetrievalQA




# Load and process documents
loader = TextLoader("vamsi_krishna.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
docs = text_splitter.split_documents(documents)

# Create embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Create FAISS vector store and save to disk
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")  # SAVES EMBEDDINGS TO LOCAL DISK

# Load from local storage (persistence)
persisted_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = persisted_vectorstore.as_retriever()

# Now you can search the embeddings
query = "What is Technical Skills vamsi has ?"
results = retriever.get_relevant_documents(query)
print(f"Found {len(results)} relevant documents")
for i, doc in enumerate(results):
    print(f"Document {i+1}: {doc.page_content[:200]}...")