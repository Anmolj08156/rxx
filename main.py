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
from typing import List, Optional, Dict

# Langchain imports for RAG functionality
from langchain_community.document.loaders import PyPDFLoader, UnstructuredWordDocumentLoader
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
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains. **Uses only pre-loaded policy.pdf for speed.**", # UPDATED DESCRIPTION
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Global RAG Chain Components ---
default_vector_store: Optional[FAISS] = None
default_qa_chain: Optional[RetrievalQA] = None
# This will track the key used for the default chain to trigger rebuilds on rotation
default_chain_initialized_with_key: Optional[str] = None 

# Dynamic document caches are no longer needed, as we're not processing dynamic URLs
# dynamic_vector_store_cache: Dict[str, FAISS] = {}
# dynamic_documents_content_cache: Dict[str, List] = {}


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

Example Question: I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?
Example Answer: Yes, if an insured person is covered under more than one policy from the same or different insurers, they have the right to choose which policy to claim from. If the Sum Insured of a single policy is exhausted, the insured person has the right to claim the balance amount from the other policy, provided the total claim amount does not exceed the total medical expenses.

Example Question: What is the ideal spark plug gap recommended?
Example Answer: The recommended spark plug gap is 0.8-0.9 mm.

Example Question: Does this come in tubeless tyre version?
Example Answer: The Super Splendor motorcycle comes with tubeless tires, 80/100-18 M/C 47P for the front and 90/90-18 M/C 51P for the rear.

Example Question: Is it compulsory to have a disc brake?
Example Answer: Disc brakes are an available option for the Super Splendor. The front brake is either a 240 mm disc or a 130 mm drum, and the rear brake is a 130 mm drum. It is not stated as compulsory.

Example Question: Can I put thums up instead of oil?
Example Answer: No, you must not use anything other than the recommended engine oil. Using anything else like Thums Up instead of engine oil will cause severe damage to the engine and is not recommended.

Example Question: Give me JS code to generate a random number between 1 and 100
Example Answer: I cannot provide JavaScript code as my function is to answer questions based on the provided documents, which are related to policy and technical specifications, not programming.

Example Question: Is Non-infective Arthritis covered?
Example Answer: Non-infective arthritis is generally excluded under Family Medicare Policy if it occurs during the first two years of continuous policy coverage, unless it arises from an accident. Specific diseases listed with a two-year waiting period include arthritis, rheumatism, gout, and spinal disorders.

Example Question: I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?
Example Answer: Under the Family Medicare Policy, Hydrocele is subject to a waiting period of two consecutive years from the first policy inception. Even if you've been a customer for 6 years, if there was any break in policy or if the specific condition had a two-year waiting period, it may still apply.

Example Question: Is abortion covered?
Example Answer: Under the Family Medicare Policy, expenses related to abortion are excluded, unless it is a result of an accident or is a medically necessary termination required to save the life of the mother.

Example Question: What is the official name of India according to Article 1 of the Constitution?
Example Answer: According to Article 1 of the Constitution of India, the official name of India is "India, that is Bharat".

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
    # 'documents' field is kept for API compatibility but will be ignored for processing.
    documents: Optional[str] = None 
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str] # List of strings, as desired

# --- Helper function for dynamic document loading ---
# This function will no longer be called during request processing.
# Kept here as a stub in case any other part of the code implicitly refers to it.
# Its purpose of caching is no longer relevant as dynamic URLs are ignored.
async def _fetch_and_load_document_from_url(url: str):
    print(f"WARNING: _fetch_and_load_document_from_url was called for URL: {url}, but dynamic URLs are being ignored for processing.")
    # Return dummy data or raise an error if this should never be called
    raise NotImplementedError("Dynamic URL processing is disabled for this service to prioritize speed.")


# --- Application Startup Event (for default policy.pdf) ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components for the default 'policy.pdf'
    once when the FastAPI application starts.
    """
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

    print("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    if not os.path.exists(PDF_PATH):
        print(f"ERROR: Default '{PDF_PATH}' not found. The API cannot function without this document as dynamic URLs are ignored.")
        # If the critical document is missing and dynamic URLs are ignored, the app must fail.
        raise RuntimeError(f"Required document '{PDF_PATH}' not found. Cannot start RAG service.")

    if FAISS is None:
        print("ERROR: FAISS is not installed. Default RAG system cannot be initialized.")
        raise RuntimeError("FAISS library not installed. Cannot start RAG service.")

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
        # Store the key it was initialized with
        default_chain_initialized_with_key = current_mistral_api_key
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your MISTRAL_API_KEYS are correct, 'policy.pdf' exists, and all required packages (like faiss-cpu, langchain-mistralai) are installed.")
        raise RuntimeError(f"RAG system initialization failed: {e}") # Raise a fatal error on startup if default fails


# --- API Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Policy Documents"
)
async def run_submission(request_body: QueryRequest):
    """
    Processes a list of natural language questions against the pre-loaded 'policy.pdf' document only.
    Any 'documents' URL provided in the request body will be ignored.
    """
    # Declare global variables used in this function *at the very top*
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

    # current_vector_store and temp_doc_path_to_clean are no longer needed as dynamic processing is removed.
    # current_vector_store = None
    # temp_doc_path_to_clean = None 

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if FAISS is None: # Still needed as a general check
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")
    
    if default_qa_chain is None: # Crucial check since we only use default
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. 'policy.pdf' might be missing or there were startup errors. Check server logs."
        )

    answers = []
    
    # Retry logic for each question with API key rotation
    max_retries_per_question = len(MISTRAL_API_KEYS) * 2 # Allow retrying with each key at least twice
    for question in request_body.questions:
        print(f"Processing question: '{question}'")
        attempt = 0
        answer_found = False
        while attempt < max_retries_per_question:
            try:
                # Rebuild default_qa_chain only if the API key has changed
                if default_chain_initialized_with_key != current_mistral_api_key:
                    print(f"Re-initializing default chain with key (partial): {current_mistral_api_key[:5]}...")
                    llm_for_default = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
                    
                    # Re-assign to global default_qa_chain
                    default_qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_for_default, # Use the LLM initialized with current_mistral_api_key
                        chain_type="stuff",
                        retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}),
                        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                    )
                    # Update the tracker
                    default_chain_initialized_with_key = current_mistral_api_key 

                result = default_qa_chain.invoke({"query": question}) # ALWAYS use default_qa_chain
                answers.append(result.get("result", "I cannot answer this question based on the provided documents."))
                answer_found = True
                break # Exit retry loop for this question if successful

            except (MistralAPIException, requests.exceptions.RequestException) as e:
                attempt += 1
                print(f"API error for current key (attempt {attempt}/{max_retries_per_question}) for question '{question}': {e}")
                if attempt < max_retries_per_question:
                    get_next_api_key() # Rotate key
                    print("Retrying with new key after a short delay...")
                    time.sleep(5) # Small delay before retrying
                else:
                    print("All API keys exhausted or max retries reached for question. Failing this question.")
                    answers.append(f"Could not retrieve an answer due to API quota limits being exceeded or persistent network issues.")
                    break # Exit retry loop, all keys exhausted for this question
            except Exception as e:
                print(f"ERROR: Failed to process question '{question}' with RAG chain (unexpected error): {e}")
                answers.append(f"An unexpected error occurred: {e}")
                answer_found = True # Treat as answered (with error message) to avoid infinite loop
                break # Exit retry loop
            
            if not answer_found and attempt >= max_retries_per_question:
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
