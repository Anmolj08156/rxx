import os
import requests
from dotenv import load_dotenv
import uuid # For unique filenames for temporary files
from pathlib import Path # For better path handling

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional

# Langchain imports for RAG functionality
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
# Attempt to import FAISS, handle ImportError if not installed
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None # Set to None if import fails, handle this later
    print("WARNING: FAISS package not found. Please install 'faiss-cpu' or 'faiss-gpu' to enable vector store functionality.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # For prompt engineering

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration & Setup ---

# Retrieve API Bearer Token from environment variables
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your .env file or Render environment.")

# Google API Key is essential for Google Gemini models
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please add it to your .env file or Render environment.")

# Define the path to the merged PDF document (fallback/initial document)
PDF_PATH = "policy.pdf" # This PDF is expected to be in the root of the project directory

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Google Gemini)",
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Global RAG Chain Components ---
# These will be initialized once on application startup for efficiency
# qa_chain: Optional[RetrievalQA] = None # Will be created per request if dynamic URL is used
# vector_store: Optional[FAISS] = None # Will be created per request if dynamic URL is used

# Global (cached) components for the default policy.pdf if no URL is provided
default_vector_store: Optional[FAISS] = None
default_qa_chain: Optional[RetrievalQA] = None


# --- Prompt Engineering ---
# This custom prompt will guide the LLM to extract precise information and explain its reasoning.
PROMPT_TEMPLATE = """
You are an expert policy document analyst. Your task is to answer user questions truthfully and based solely on the provided context.
The context provided comes from official policy documents, contracts, or emails.
If the answer is not available in the provided context, state "I cannot answer this question based on the provided documents."
Do not make up information.
When providing an answer, aim for conciseness and accuracy, directly quoting or paraphrasing relevant clauses where appropriate.
If the question asks for a specific detail (e.g., a number, a period), extract that exact detail.

Context:
{context}

Question: {question}
Answer:
"""
CUSTOM_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


# --- API Authentication Dependency ---
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the bearer token for API authentication.
    """
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    # The 'documents' field is now used to optionally provide a URL to a document
    documents: Optional[str] = None # URL to a PDF, Word doc, or other supported type
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Helper function for dynamic document loading ---
async def _fetch_and_load_document_from_url(url: str):
    """
    Fetches a document from a given URL, saves it temporarily, and loads it.
    Supports PDF and DOCX based on file extension.
    """
    temp_dir = Path("./temp_docs") # Create a temporary directory
    temp_dir.mkdir(exist_ok=True)

    file_extension = url.split('.')[-1].split('?')[0].lower()
    temp_file_name = f"temp_doc_{uuid.uuid4()}.{file_extension}"
    temp_file_path = temp_dir / temp_file_name

    print(f"Fetching document from URL: {url} to {temp_file_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Document saved temporarily: {temp_file_path}")

        if file_extension == "pdf":
            loader = PyPDFLoader(str(temp_file_path))
        elif file_extension in ["doc", "docx"]:
            loader = UnstructuredWordDocumentLoader(str(temp_file_path))
        else:
            raise ValueError(f"Unsupported document type from URL: {file_extension}. Only PDF, DOC, DOCX are supported.")

        documents = loader.load()
        print(f"Loaded {len(documents)} pages/documents from URL.")
        return documents, temp_file_path # Return path for cleanup

    except requests.exceptions.RequestException as req_e:
        print(f"ERROR: Failed to fetch document from URL {url}: {req_e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Failed to fetch document from URL: {req_e}")
    except Exception as e:
        print(f"ERROR: Failed to load document from {temp_file_path}: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Error processing document from URL: {e}")


# --- Application Startup Event (for default policy.pdf) ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components for the default 'policy.pdf'
    once when the FastAPI application starts.
    """
    global default_qa_chain, default_vector_store

    print("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    # 1. Verify PDF file existence for default
    if not os.path.exists(PDF_PATH):
        print(f"WARNING: Default '{PDF_PATH}' not found. The API will only work if documents are provided via URL in requests.")
        return # Do not raise, allow dynamic loading

    # 2. Check if FAISS is available before proceeding
    if FAISS is None:
        print("ERROR: FAISS is not installed. Default RAG system cannot be initialized.")
        return

    try:
        # 3. Load the default document
        print(f"Loading default document from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {PDF_PATH}")

        # 4. Split documents into chunks
        print("Splitting default documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        print(f"Created {len(docs)} text chunks for default policy.")

        # 5. Create embeddings and build FAISS vector store for default
        print("Creating embeddings and building default FAISS vector store (this may take a moment)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        print("Default FAISS vector store built successfully.")

        # 6. Initialize the ChatGoogleGenerativeAI LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
        
        # 7. Create the RetrievalQA chain for default
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Apply custom prompt here
        )
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your GOOGLE_API_KEY is correct, 'policy.pdf' exists, and all required packages (like faiss-cpu) are installed.")
        # Do not raise here, allow the app to start to handle dynamic document uploads
        # raise


# --- API Endpoint ---
@app.post(
    "/hackrx/run", # Endpoint path as specified in the problem statement
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)], # Apply authentication to this endpoint
    summary="Run LLM-Powered Query-Retrieval on Policy Documents"
)
async def run_submission(request_body: QueryRequest):
    """
    Processes a list of natural language questions against the provided document(s) (URL or default)
    and returns contextual answers.

    If 'documents' URL is provided, it will dynamically load and process that document.
    Otherwise, it will use the pre-loaded 'policy.pdf'.
    """
    current_vector_store = None
    current_qa_chain = None
    temp_doc_path = None # To store path of dynamically fetched document for cleanup

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    try:
        # Determine which document source to use
        if request_body.documents:
            print(f"Processing request with dynamic document URL: {request_body.documents}")
            # Fetch and load the document from the URL
            documents, temp_doc_path = await _fetch_and_load_document_from_url(request_body.documents)

            # Split documents into chunks for the dynamic document
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            print(f"Created {len(docs)} text chunks for dynamic document.")

            # Create embeddings and build FAISS vector store for the dynamic document
            if FAISS is None:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="FAISS is not installed, cannot process dynamic documents.")
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            current_vector_store = FAISS.from_documents(docs, embeddings)
            print("FAISS vector store built for dynamic document.")

            # Initialize the ChatGoogleGenerativeAI LLM for the dynamic chain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
            
            # Create the RetrievalQA chain for the dynamic document
            current_qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=current_vector_store.as_retriever(),
                chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Apply custom prompt here
            )
            print("RetrievalQA chain initialized for dynamic document.")

        else:
            print(f"Processing request with default document: {PDF_PATH}")
            # Use the globally pre-initialized QA chain from default policy.pdf
            if default_qa_chain is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG system (default document) is not initialized. Please check server logs for startup errors."
                )
            current_qa_chain = default_qa_chain

        answers = []
        for question in request_body.questions:
            print(f"Processing question: '{question}'")
            # Invoke the QA chain with the user's question
            result = current_qa_chain.invoke({"query": question})
            answers.append(result.get("result", "Could not retrieve an answer."))
            print(f"Answer generated for '{question}'.")

        print("--- All questions processed. Sending response. ---")
        return {"answers": answers}

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        # Clean up temporary document if one was downloaded
        if temp_doc_path and temp_doc_path.exists():
            try:
                os.remove(temp_doc_path)
                print(f"Cleaned up temporary file: {temp_doc_path}")
                # Also remove the temp_docs directory if it's empty
                if not any(Path("./temp_docs").iterdir()):
                    os.rmdir("./temp_docs")
            except Exception as cleanup_e:
                print(f"WARNING: Failed to clean up temporary file {temp_doc_path}: {cleanup_e}")


# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
