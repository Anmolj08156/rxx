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
from typing import List, Optional, Dict # ADDED Dict HERE

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
# NEW GLOBAL VARIABLE to track the key used for the default chain
default_chain_initialized_with_key: Optional[str] = None 

# Cache for dynamic vector stores to avoid re-processing same URL multiple times within a single instance's lifetime
dynamic_vector_store_cache: Dict[str, FAISS] = {}
dynamic_documents_content_cache: Dict[str, List] = {} # Cache raw documents to avoid re-loading from disk after first parse


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
    documents: Optional[str] = None # URL to a PDF, Word doc, or other supported type
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str] # List of strings, as desired

# --- Helper function for dynamic document loading ---
# Cache for dynamic vector stores to avoid re-processing same URL multiple times within a single instance's lifetime
dynamic_vector_store_cache: Dict[str, FAISS] = {}
dynamic_documents_content_cache: Dict[str, List] = {} # Cache raw documents to avoid re-loading from disk after first parse


async def _fetch_and_load_document_from_url(url: str):
    """
    Fetches a document from a given URL, saves it temporarily, and loads it.
    Supports PDF and DOCX based on file extension.
    Caches the loaded documents to avoid repeated file operations within a request.
    """
    # Check if document content is already in cache (prevents re-reading from temp file if just processed)
    if url in dynamic_documents_content_cache:
        print(f"Using cached document content for URL: {url}")
        return dynamic_documents_content_cache[url], None # No temp path to clean up as it's from cache

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
        dynamic_documents_content_cache[url] = documents # Cache the loaded documents
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
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

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
        # Store the key it was initialized with
        default_chain_initialized_with_key = current_mistral_api_key
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
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

    current_vector_store = None
    current_qa_chain = None
    temp_doc_path_to_clean = None # Track temp path for final cleanup

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")

    answers = []
    
    try:
        # --- CRITICAL PERFORMANCE OPTIMIZATION FOR DYNAMIC DOCUMENTS ---
        # Load and process dynamic document ONCE per request, not per question
        if request_body.documents:
            cached_vector_store = dynamic_vector_store_cache.get(request_body.documents)
            if cached_vector_store:
                print(f"Using cached FAISS vector store for URL: {request_body.documents}")
                current_vector_store = cached_vector_store
                # No temp_doc_path_to_clean if using cache
            else:
                print(f"Processing NEW dynamic document URL: {request_body.documents}")
                documents, temp_doc_path = await _fetch_and_load_document_from_url(request_body.documents)
                temp_doc_path_to_clean = temp_doc_path # Store for cleanup

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                docs = text_splitter.split_documents(documents)
                print(f"Created {len(docs)} text chunks for dynamic document.")

                # Use current_mistral_api_key for embeddings (might be rotated)
                embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)
                current_vector_store = FAISS.from_documents(docs, embeddings)
                dynamic_vector_store_cache[request_body.documents] = current_vector_store # Cache the new vector store
                print("FAISS vector store built and cached for dynamic document.")
            
            # Initialize LLM with the current API key (might be rotated)
            llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
            current_qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=current_vector_store.as_retriever(search_kwargs={"k": 4}), # Reduced k
                chain_type_kwargs={"prompt": CUSTOM_PROMPT}
            )
            print("RetrievalQA chain initialized for dynamic document.")
        else:
            print(f"Processing request with default document: {PDF_PATH}")
            if default_qa_chain is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Default RAG system is not initialized. 'policy.pdf' might be missing or there were startup errors."
                )
            # Rebuild default_qa_chain only if the API key has changed
            if default_chain_initialized_with_key != current_mistral_api_key:
                 print(f"Re-initializing default chain with key (partial): {current_mistral_api_key[:5]}...")
                 llm_for_default = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
                 default_qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_for_default, # Use the LLM initialized with current_mistral_api_key
                    chain_type="stuff",
                    retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}),
                    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                )
                 default_chain_initialized_with_key = current_mistral_api_key # Update tracker
            current_qa_chain = default_qa_chain

        # --- Process each question using the ONE prepared QA chain for this request ---
        max_retries_per_question = len(MISTRAL_API_KEYS) * 2 # Allow retrying with each key at least twice
        for question in request_body.questions:
            print(f"Processing question: '{question}'")
            attempt = 0
            answer_found = False
            while attempt < max_retries_per_question:
                try:
                    # For a retry within a question, ensure current_qa_chain's LLM uses current_mistral_api_key
                    # This is done by dynamically setting the LLM on the chain object if needed
                    # (Note: Some LangChain versions might require rebuilding chain for LLM change)
                    # To be safe, we rebuild the current_qa_chain on retry if the key changed.
                    if current_qa_chain.llm.mistral_api_key != current_mistral_api_key:
                        llm_for_retry = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
                        current_qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_for_retry,
                            chain_type="stuff",
                            retriever=current_vector_store.as_retriever(search_kwargs={"k": 4}),
                            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                        )

                    result = current_qa_chain.invoke({"query": question})
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

    except HTTPException:
        raise # Re-raise HTTPExceptions directly
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during overall query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        # Clean up temporary document if one was downloaded in this request
        if temp_doc_path_to_clean and temp_doc_path_to_clean.exists():
            try:
                os.remove(temp_doc_path_to_clean)
                print(f"Cleaned up temporary file: {temp_doc_path_to_clean}")
                # Clean up temp_docs directory if it becomes empty
                if Path("./temp_docs").exists() and not any(Path("./temp_docs").iterdir()):
                    os.rmdir("./temp_docs")
            except Exception as cleanup_e:
                print(f"WARNING: Failed to clean up temporary file {temp_doc_path_to_clean}: {cleanup_e}")

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
