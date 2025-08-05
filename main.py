import os
import requests
from dotenv import load_dotenv
import uuid
from pathlib import Path
import time
import itertools
import logging
import json # Import json for logging request body
import random # Import random for jitter in backoff
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None
    print("WARNING: FAISS package not found. Please install 'faiss-cpu' or 'faiss-gpu' to enable vector store functionality.")

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
    """
    Callback for tenacity to rotate the API key on a retry attempt.
    """
    global current_mistral_api_key
    old_key_partial = current_mistral_api_key[:5] + "..."
    current_mistral_api_key = next(api_key_iterator)
    new_key_partial = current_mistral_api_key[:5] + "..."
    logger.warning(
        f"API call failed (status {retry_state.outcome.result().response.status_code}). "
        f"Rotating key from {old_key_partial} to {new_key_partial}. "
        f"Attempt {retry_state.attempt_number + 1} of {retry_state.retry_object.stop.max_attempts}."
    )
    # Re-initialize the LLM chain with the new key in the calling function
    return current_mistral_api_key

PDF_PATH = "policy.pdf"

app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Mistral AI)",
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains. **Uses only pre-loaded policy.pdf for speed.**",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

default_vector_store: Optional[FAISS] = None
# The QA chain will be initialized per-request to ensure the current API key is used.
default_qa_chain: Optional[RetrievalQA] = None 
default_chain_initialized_with_key: Optional[str] = None

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

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    global default_vector_store

    logger.info("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    if not os.path.exists(PDF_PATH):
        logger.error(f"ERROR: Default '{PDF_PATH}' not found. The API cannot function without this document.")
        raise RuntimeError(f"Required document '{PDF_PATH}' not found. Cannot start RAG service.")

    if FAISS is None:
        logger.error("ERROR: FAISS is not installed. Default RAG system cannot be initialized.")
        raise RuntimeError("FAISS library not installed. Cannot start RAG service.")

    try:
        logger.info(f"Loading default document from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {PDF_PATH}")

        logger.info("Splitting default documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} text chunks for default policy.")

        logger.info("Creating embeddings and building default FAISS vector store (this may take a moment)...")
        # Use a temporary LLM instance for embeddings during startup with the first key
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        logger.info("Default FAISS vector store built successfully.")
        logger.info("API is ready to receive requests.")

    except Exception as e:
        logger.exception(f"--- ERROR during Default RAG System Initialization: {e} ---")
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
    global default_vector_store, current_mistral_api_key

    try:
        raw_body = request.state.body
        if not raw_body:
            raise ValueError("Request body is empty.")
        request_body = QueryRequest.parse_raw(raw_body)
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request body format or missing data.")

    logger.info(f"Processing request for {len(request_body.questions)} questions.")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed.")
    if default_vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. 'policy.pdf' might be missing or there were startup errors."
        )

    answers = []
    for question in request_body.questions:
        try:
            # Use tenacity to handle retries and API key rotation
            answer = await process_question_with_retries(question, default_vector_store)
            answers.append(answer)
        except httpx.HTTPStatusError as e:
            logger.error(f"Final failure after all retries for question '{question}': {e}")
            answers.append(f"Could not retrieve an answer due to API quota limits or network issues. Error: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing question '{question}': {e}")
            answers.append(f"An unexpected internal error occurred: {e}")
            
    logger.info("--- All questions processed. Sending response. ---")
    return {"answers": answers}

@retry(
    stop=stop_after_attempt(len(MISTRAL_API_KEYS)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=get_next_api_key
)
async def process_question_with_retries(question: str, vector_store: FAISS) -> str:
    """
    Handles the RAG chain invocation with built-in retries and key rotation.
    """
    global current_mistral_api_key

    llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
    
    # Re-initialize the chain on each attempt to ensure the LLM instance with the current key is used
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    logger.info(f"Invoking RAG chain for question: '{question}' with key: {current_mistral_api_key[:5]}...")
    
    # Langchain's invoke is now used with a standard dictionary
    result = await qa_chain.ainvoke({"query": question})
    
    # Check for a specific 429 error and raise a custom exception if needed, to be caught by tenacity
    # httpx.HTTPStatusError is a more general exception, let's assume it covers 429
    
    return result.get("result", "I cannot answer this question based on the provided documents.")

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
