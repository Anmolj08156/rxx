import os
import requests
from dotenv import load_dotenv
import uuid
from pathlib import Path
import time # For retries
import itertools # For cycling through API keys

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional

# Langchain imports for RAG functionality
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None
    print("WARNING: FAISS package not found. Please install 'faiss-cpu' or 'faiss-gpu' to enable vector store functionality.")

# --- MISTRAL AI Imports ---
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
# Handle mistralai.exceptions import more robustly
try:
    from mistralai.exceptions import MistralAPIException
except ImportError:
    print("WARNING: Could not import MistralAPIException directly. Falling back to requests.exceptions.RequestException for error handling.")
    MistralAPIException = requests.exceptions.RequestException # Use a more general exception as fallback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration & Setup ---

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your .env file or Render environment.")

MISTRAL_API_KEYS_STR = os.getenv("MISTRAL_API_KEYS")
if not MISTRAL_API_KEYS_STR:
    raise ValueError("MISTRAL_API_KEYS environment variable is not set. Please add it to your .env file or Render environment.")

MISTRAL_API_KEYS = [k.strip() for k in MISTRAL_API_KEYS_STR.split(',')]
if not MISTRAL_API_KEYS:
    raise ValueError("MISTRAL_API_KEYS environment variable is set but contains no valid keys.")

api_key_iterator = itertools.cycle(MISTRAL_API_KEYS)
current_mistral_api_key = next(api_key_iterator)

def get_next_api_key():
    """Cycles to the next API key in the list."""
    global current_mistral_api_key
    current_mistral_api_key = next(api_key_iterator)
    print(f"Switched to next Mistral API Key. Current key (partial): {current_mistral_api_key[:5]}...")
    return current_mistral_api_key

# Define the path to the merged PDF document (fallback/initial document)
PDF_PATH = "policy.pdf"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Mistral AI)",
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Global RAG Chain Components ---
default_vector_store: Optional[FAISS] = None
default_qa_chain: Optional[RetrievalQA] = None


# --- Prompt Engineering with Reduced Few-Shot Examples for Speed ---
PROMPT_TEMPLATE = """
You are an expert in analyzing various types of documents, including policy documents, contracts, legal texts, and technical manuals.
Your task is to answer user queries accurately, concisely, and comprehensively, based **only** on the provided context.
If the exact answer or sufficient information is not found in the context, state: "I cannot answer this question based on the provided documents."
Do not generate information that is not supported by the context.

CRITICAL INSTRUCTIONS:
- Answer in EXACTLY 2-3 lines maximum (not paragraphs).
- Include specific numbers, amounts, percentages, and timeframes if relevant.
- Be direct and factual - no filler words or explanations.
- If multiple related points exist, combine them in one coherent response.
- Start directly with the answer - no introductory phrases.
- Do NOT include any source citations in your answers.

BEGIN EXAMPLES:

Example Question: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
Example Answer: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Example Question: What is the waiting period for pre-existing diseases (PED) to be covered?
Example Answer: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

Example Question: How does the policy define a 'Hospital'?
Example Answer: A Hospital is an institution with at least 10 Inpatient beds (towns < 10 lakhs population) or 15 Inpatient beds (other places), qualified 24/7 nursing staff and medical practitioners, equipped operation theatre, and daily patient records.

Example Question: Is abortion covered?
Example Answer: Expenses related to abortion are excluded, unless it is a result of an accident or is a medically necessary termination required to save the life of the mother.

Example Question: What is the official name of India according to Article 1 of the Constitution?
Example Answer: According to Article 1 of the Constitution of India, the official name of India is "India, that is Bharat".

Example Question: What is abolished by Article 17 of the Constitution?
Example Answer: Article 17 of the Constitution abolishes "Untouchability" and its practice in any form is forbidden.

Example Question: According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?
Example Answer: According to Article 24 of the Constitution, no child below the age of fourteen years shall be employed to work in any factory or mine or engaged in any other hazardous employment.

Example Question: If my car is stolen, what case will it be in law?
Example Answer: If your car is stolen, it would typically be considered a criminal case under the Indian Penal Code, specifically dealing with theft (Section 378 and related provisions).

Example Question: If an insured person takes treatment for arthritis at home because no hospital beds are available, under what circumstances would these expenses NOT be covered, even if a doctor declares the treatment was medically required?
Example Answer: Home nursing expenses are generally excluded unless specifically allowed and meet strict criteria, usually requiring prior hospitalization for the same serious illness/injury and a medical practitioner's certification.

Example Question: A claim was lodged for expenses on a prosthetic device after a hip replacement surgery. The hospital bill also includes the cost of a walker and a lumbar belt post-discharge. Which items are payable?
Example Answer: While the cost of the prosthetic device is covered, expenses for items such as walking aids (like walkers) and belts (like lumbar belts) are typically excluded as non-medical or not integral to the procedure.

END EXAMPLES

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
    documents: Optional[str] = None # URL to a PDF, Word doc, or other supported type
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str] # List of strings, as desired

# --- Helper function for dynamic document loading ---
async def _fetch_and_load_document_from_url(url: str):
    """
    Fetches a document from a given URL, saves it temporarily, and loads it.
    Supports PDF and DOCX based on file extension.
    """
    temp_dir = Path("./temp_docs")
    temp_dir.mkdir(exist_ok=True)

    file_extension = url.split('.')[-1].split('?')[0].lower()
    temp_file_name = f"temp_doc_{uuid.uuid4()}.{file_extension}"
    temp_file_path = temp_dir / temp_file_name

    print(f"Fetching document from URL: {url} to {temp_file_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

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
        return documents, temp_file_path

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
    global default_qa_chain, default_vector_store, current_mistral_api_key

    print("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    if not os.path.exists(PDF_PATH):
        print(f"WARNING: Default '{PDF_PATH}' not found. The API will only work if documents are provided via URL in requests.")
        return

    if FAISS is None:
        print("ERROR: FAISS is not installed. Default RAG system cannot be initialized.")
        return

    try:
        print(f"Loading default document from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {PDF_PATH}")

        print("Splitting default documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        print(f"Created {len(docs)} text chunks for default policy.")

        print("Creating embeddings and building default FAISS vector store (this may take a moment)...")
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        print("Default FAISS vector store built successfully.")

        llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
        
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}), # Reduced k to 4
            chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Apply custom prompt here
        )
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your MISTRAL_API_KEYS are correct, 'policy.pdf' exists, and all required packages (like faiss-cpu, langchain-mistralai) are installed.")
        # Do not raise here, allow the app to start to handle dynamic document uploads


# --- API Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Policy Documents"
)
async def run_submission(request_body: QueryRequest):
    """
    Processes a list of natural language questions against the provided document(s) (URL or default)
    and returns contextual answers.
    """
    # Declare global variables used in this function
    global default_qa_chain, default_vector_store, current_mistral_api_key

    current_vector_store = None
    current_qa_chain = None
    temp_doc_path = None

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")

    answers = []
    
    # Retry logic for each question with API key rotation
    max_retries = len(MISTRAL_API_KEYS) * 2 # Allow retrying with each key at least twice
    for question in request_body.questions:
        print(f"Processing question: '{question}'")
        attempt = 0
        answer_found = False
        while attempt < max_retries:
            try:
                # Re-initialize LLM and Embeddings with the current_mistral_api_key for each attempt/rotation
                llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
                embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)

                if request_body.documents:
                    print(f"Re-initializing chain for dynamic document with key (partial): {current_mistral_api_key[:5]}...")
                    documents, temp_doc_path = await _fetch_and_load_document_from_url(request_body.documents)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    docs = text_splitter.split_documents(documents)
                    temp_vector_store = FAISS.from_documents(docs, embeddings)
                    current_qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=temp_vector_store.as_retriever(search_kwargs={"k": 4}), # Reduced k
                        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                    )
                else:
                    # For the default chain, if the key was just rotated, we need to rebuild it
                    # to ensure it uses the new key for its internal LLM and Embeddings.
                    # Note: default_vector_store's embeddings are only set once at startup.
                    if default_qa_chain is None or (hasattr(default_qa_chain, 'llm') and hasattr(default_qa_chain.llm, 'mistral_api_key') and default_qa_chain.llm.mistral_api_key != current_mistral_api_key):
                        print(f"Re-initializing default chain with key (partial): {current_mistral_api_key[:5]}...")
                        default_qa_chain = RetrievalQA.from_chain_type(
                            llm=llm, # Use the LLM initialized with current_mistral_api_key
                            chain_type="stuff",
                            retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}), # Reduced k
                            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                        )
                    current_qa_chain = default_qa_chain


                result = current_qa_chain.invoke({"query": question})
                answers.append(result.get("result", "I cannot answer this question based on the provided documents."))
                answer_found = True
                break # Exit retry loop if successful

            except (MistralAPIException, requests.exceptions.RequestException) as e: # Catch both specific and general
                attempt += 1
                print(f"API error for current key (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    get_next_api_key() # Rotate key
                    print("Retrying with new key after a short delay...")
                    time.sleep(5) # Small delay before retrying
                else:
                    print("All API keys exhausted or max retries reached. Failing this question.")
                    answers.append(f"Could not retrieve an answer due to API quota limits being exceeded or persistent network issues.")
                    break # Exit retry loop, all keys exhausted for this question
            except Exception as e:
                print(f"ERROR: Failed to process question '{question}' with RAG chain (unexpected error): {e}")
                answers.append(f"An unexpected error occurred: {e}")
                answer_found = True
                break
            
        if not answer_found and attempt >= max_retries:
            answers.append(f"Failed to answer '{question}' after multiple retries due to persistent API quota limits or other issues.")

        print(f"Answer generated for '{question}'.")

    print("--- All questions processed. Sending response. ---")
    return {"answers": answers}

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
