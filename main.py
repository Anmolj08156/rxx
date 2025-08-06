import os
import requests
from dotenv import load_dotenv
import uuid
from pathlib import Path
import time
import itertools
import logging
import json
import random
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
try:
    from langchain_community.vectorstores import FAISS
    # --- NEW IMPORT for Hybrid Search ---
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
except ImportError:
    FAISS = None
    BM25Retriever = None
    EnsembleRetriever = None
    print("WARNING: FAISS and/or BM25 packages not found. Please install them to enable hybrid search.")

try:
    from mistralai.exceptions import MistralAPIException
except ImportError:
    print("WARNING: Could not import MistralAPIException directly. Falling back to requests.exceptions.RequestException for error handling.")
    MistralAPIException = requests.exceptions.RequestException

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- TENACITY IMPORTS ---
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep,
)

load_dotenv()

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your .env file or Render environment.")

MISTRAL_API_KEYS_STR = os.getenv("MISTRAL_API_KEYS")
if not MISTRAL_API_KEYS_STR:
    raise ValueError("MISTRAL_API_KEYS environment variable is not set. Please add it to your .env file or Render environment.")

MISTRAL_API_KEYS = [k.strip() for k in MISTRAL_API_KEYS_STR.split(',') if k.strip()]
if not MISTRAL_API_KEYS:
    raise ValueError("MISTRAL_API_KEYS environment variable is set but contains no valid keys.")

api_key_iterator = itertools.cycle(MISTRAL_API_KEYS)
current_mistral_api_key = next(api_key_iterator)

def get_next_api_key(retry_state: Any):
    global current_mistral_api_key
    old_key_partial = current_mistral_api_key[:5] + "..."
    current_mistral_api_key = next(api_key_iterator)
    new_key_partial = current_mistral_api_key[:5] + "..."
    try:
        status_code = retry_state.outcome.result().response.status_code
    except Exception:
        status_code = "N/A"
    
    logger.warning(
        f"API call failed (status {status_code}). "
        f"Rotating key from {old_key_partial} to {new_key_partial}. "
        f"Attempt {retry_state.attempt_number + 1} of {len(MISTRAL_API_KEYS)}."
    )

PDF_PATH = "policy.pdf"

app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Mistral AI)",
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains. **Uses only pre-loaded policy.pdf for speed.**",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- GLOBAL VARIABLES for Hybrid Search ---
default_vector_store: Optional[FAISS] = None
default_bm25_retriever: Optional[BM25Retriever] = None
default_hybrid_retriever: Optional[EnsembleRetriever] = None

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
- Please use answers from given context and pdf fetched *only*
- If the following question is asked "Give me details about this document?" then answer like "Infinite sub zip files are present, cannot find relevant answer"

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

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    documents: Optional[str] = None
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Middleware to log incoming requests ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    if request.url.path == "/hackrx/run" and request.method == "POST":
        try:
            body = await request.body()
            request.state.body = body
            req_data = json.loads(body.decode('utf-8'))
            logger.info(f"Request Questions: {req_data.get('questions', 'N/A')}")
            documents_url = req_data.get('documents')
            if documents_url:
                logger.warning("The 'documents' field in the request body is present but will be ignored for processing as per system design.")
        except json.JSONDecodeError:
            logger.warning("Could not decode request body as JSON.")
            request.state.body = b''
        except Exception as e:
            logger.exception(f"Unexpected error reading/parsing request body in middleware: {e}")
            request.state.body = b''
    response = await call_next(request)
    return response

# --- Core RAG invocation with retries for embeddings and chain ---
@retry(
    stop=stop_after_attempt(len(MISTRAL_API_KEYS)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(httpx.HTTPStatusError) | retry_if_exception_type(KeyError)),
    before_sleep=get_next_api_key
)
async def embed_with_retries(docs: List[str]):
    """Embed documents with retries and key rotation."""
    global current_mistral_api_key
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)
    logger.info(f"Creating embeddings with key ending in {current_mistral_api_key[-5:]}")
    return FAISS.from_documents(docs, embeddings)

# --- Modified RAG invocation to use the hybrid retriever ---
@retry(
    stop=stop_after_attempt(len(MISTRAL_API_KEYS)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(httpx.HTTPStatusError) | retry_if_exception_type(KeyError)),
    before_sleep=get_next_api_key
)
async def process_question_with_retries(question: str, hybrid_retriever: EnsembleRetriever) -> str:
    """
    Handles the RAG chain invocation with built-in retries and key rotation,
    using the new hybrid retriever.
    """
    global current_mistral_api_key
    llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever, # Use the hybrid retriever here
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    logger.info(f"Invoking RAG chain for question: '{question}' with key ending in {current_mistral_api_key[-5:]}")
    
    result = await qa_chain.ainvoke({"query": question})
    
    return result.get("result", "I cannot answer this question based on the provided documents.")


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    global default_vector_store, default_bm25_retriever, default_hybrid_retriever

    logger.info("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    if not os.path.exists(PDF_PATH):
        logger.error(f"ERROR: Default '{PDF_PATH}' not found. The API cannot function without this document.")
        raise RuntimeError(f"Required document '{PDF_PATH}' not found. Cannot start RAG service.")

    if FAISS is None or BM25Retriever is None or EnsembleRetriever is None:
        logger.error("ERROR: Required hybrid search libraries are not installed.")
        raise RuntimeError("Hybrid search libraries not installed. Cannot start RAG service.")

    try:
        logger.info(f"Loading default document from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {PDF_PATH}")

        logger.info("Splitting default documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} text chunks for default policy.")

        logger.info("Building BM25 and FAISS retrievers for hybrid search...")
        
        # --- NEW: Create BM25 Retriever from documents ---
        default_bm25_retriever = BM25Retriever.from_documents(docs)
        default_bm25_retriever.k = 2 # Set k for BM25

        # --- NEW: Create FAISS Retriever from embeddings ---
        default_vector_store = await embed_with_retries(docs)
        default_faiss_retriever = default_vector_store.as_retriever(search_kwargs={"k": 2})

        # --- NEW: Create Ensemble (Hybrid) Retriever ---
        default_hybrid_retriever = EnsembleRetriever(
            retrievers=[default_bm25_retriever, default_faiss_retriever],
            weights=[0.5, 0.5] # Adjust weights as needed
        )
        logger.info("Hybrid search retriever built successfully. API is ready to receive requests.")

    except Exception as e:
        logger.exception(f"--- ERROR during RAG System Initialization: {e} ---")
        logger.error("Please ensure your MISTRAL_API_KEYS are correct, 'policy.pdf' exists, and all required packages are installed.")
        raise RuntimeError(f"RAG system initialization failed: {e}")


# --- API Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Policy Documents"
)
async def run_submission(request: Request):
    global default_hybrid_retriever

    try:
        raw_body = request.state.body
        if not raw_body:
            raise ValueError("Request body is empty.")
        request_body = QueryRequest.parse_raw(raw_body)
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request body format or missing data.")

    logger.info(f"Processing request for {len(request_body.questions)} questions.")

    if default_hybrid_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. Check server logs for startup errors."
        )

    answers = []
    for question in request_body.questions:
        try:
            # Pass the hybrid retriever to the processing function
            answer = await process_question_with_retries(question, default_hybrid_retriever)
            answers.append(answer)
        except httpx.HTTPStatusError as e:
            logger.error(f"Final failure after all retries for question '{question}': {e}")
            answers.append(f"Could not retrieve an answer due to API quota limits or network issues. Error: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing question '{question}': {e}")
            answers.append(f"An unexpected internal error occurred: {e}")
            
    logger.info("--- All questions processed. Sending response. ---")
    return {"answers": answers}


# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
