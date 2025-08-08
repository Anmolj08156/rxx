import os
import itertools
import logging
import json
import httpx
import asyncio
import re
import io

from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None
    print("WARNING: FAISS package not found. Please install 'faiss-cpu' or 'faiss-gpu' to enable vector store functionality.")

# --- NEW IMPORTS for Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# --- TENACITY IMPORTS ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep,
)

# --- CONFIGURATION & SETUP ---
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your environment.")

GOOGLE_API_KEYS_STR = os.getenv("GOOGLE_API_KEYS")
if not GOOGLE_API_KEYS_STR:
    raise ValueError("GOOGLE_API_KEYS environment variable is not set. Please add it to your environment.")

GOOGLE_API_KEYS = [k.strip() for k in GOOGLE_API_KEYS_STR.split(',') if k.strip()]
if not GOOGLE_API_KEYS:
    raise ValueError("GOOGLE_API_KEYS environment variable is set but contains no valid keys.")

api_key_iterator = itertools.cycle(GOOGLE_API_KEYS)
current_google_api_key = next(api_key_iterator)

# PDF path is now deprecated, the system is dynamic
PDF_PATH = "policy.pdf"

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Gemini)",
    description="API for processing large documents and making contextual decisions. **Now handles dynamic PDF URLs and embedded web content for puzzle-solving.**",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- GLOBAL VARIABLES & DEPENDENCIES ---
# default_vector_store is no longer used for dynamic document processing
default_vector_store: Optional[FAISS] = None

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the bearer token for API access."""
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Pydantic Models for API Request/Response
class QueryRequest(BaseModel):
    documents: str # documents is now a required URL string
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- PROMPT TEMPLATE ---
# The prompt is now generalized to handle puzzle-solving from multiple sources
PROMPT_TEMPLATE = """
You are an expert in analyzing various types of documents, including puzzles, policy documents, contracts, and web pages.
Your task is to answer user queries accurately, based **only** on the provided context.
The context may contain content from a main PDF document and other web pages referenced within it.
Your goal is to solve any puzzles presented in the questions using all of the provided context.

CRITICAL INSTRUCTIONS:
- Answer in EXACTLY 2-3 lines maximum (not paragraphs).
- Include specific numbers, amounts, percentages, and timeframes if relevant.
- Be direct and factual - no filler words or explanations.
- If multiple related points exist, combine them in one coherent response.
- Start directly with the answer - no introductory phrases.
- Do NOT include any source citations in your answers.
- Please use answers from given context *only* and treat outside context thing as incorrect.

Context:
{context}

Question: {question}
Answer:
"""
CUSTOM_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# --- UTILITY FUNCTIONS ---
def get_next_api_key(retry_state: Any):
    """Callback for tenacity to rotate the API key on a retry attempt."""
    global current_google_api_key
    old_key_partial = current_google_api_key[:5] + "..."
    current_google_api_key = next(api_key_iterator)
    new_key_partial = current_google_api_key[:5] + "..."
    
    status_code = "N/A"
    if retry_state.outcome and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
            status_code = exception.response.status_code
    
    logger.warning(
        f"API call failed (status {status_code}). "
        f"Rotating key from {old_key_partial} to {new_key_partial}. "
        f"This is attempt {retry_state.attempt_number + 1} of {len(GOOGLE_API_KEYS)}."
    )

# New function to extract URLs from a string
def extract_urls_from_string(text: str) -> List[str]:
    """Finds and returns all valid URLs in a given string."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)

# --- CORE RAG INVOCATION WITH RETRIES ---
@retry(
    stop=stop_after_attempt(len(GOOGLE_API_KEYS)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(httpx.HTTPStatusError) | retry_if_exception_type(KeyError)),
    before_sleep=get_next_api_key
)
async def embed_with_retries(docs: List[Document]):
    """Embed documents with retries and key rotation."""
    global current_google_api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=current_google_api_key)
    logger.info(f"Creating embeddings with key ending in {current_google_api_key[-5:]}")
    # Using an in-memory docstore is not required for FAISS.from_documents
    return FAISS.from_documents(docs, embeddings)

@retry(
    stop=stop_after_attempt(len(GOOGLE_API_KEYS)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(httpx.HTTPStatusError) | retry_if_exception_type(KeyError)),
    before_sleep=get_next_api_key
)
async def process_question_with_retries(question: str, vector_store: FAISS) -> str:
    """Handles the RAG chain invocation with built-in retries and key rotation."""
    global current_google_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=current_google_api_key)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    logger.info(f"Invoking RAG chain for question: '{question}' with key ending in {current_google_api_key[-5:]}")
    
    result = await qa_chain.ainvoke({"query": question})
    
    return result.get("result", "I cannot answer this question based on the provided documents.")

# --- MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming request details, including the body for specific endpoints."""
    logger.info(f"Incoming Request: {request.method} {request.url}")
    if request.url.path == "/hackrx/run" and request.method == "POST":
        try:
            body = await request.body()
            request.state.body = body
            req_data = json.loads(body.decode('utf-8'))
            logger.info(f"Request Questions: {req_data.get('questions', 'N/A')}")
            documents_url = req_data.get('documents')
            if documents_url:
                logger.info(f"The 'documents' field with URL '{documents_url}' is present and will be processed.")
        except json.JSONDecodeError:
            logger.warning("Could not decode request body as JSON.")
            request.state.body = b''
        except Exception as e:
            logger.exception(f"Unexpected error reading/parsing request body in middleware: {e}")
            request.state.body = b''
    response = await call_next(request)
    return response

# --- APPLICATION LIFECYCLE EVENTS ---
@app.on_event("startup")
async def startup_event():
    """Initializes the RAG system by loading, chunking, and embedding the policy document."""
    global default_vector_store
    
    logger.info("--- Application Startup: RAG system will now load documents dynamically from URLs. ---")
    
    if FAISS is None:
        logger.error("ERROR: FAISS is not installed. RAG system cannot be initialized.")
        raise RuntimeError("FAISS library not installed. Cannot start RAG service.")
    
    # We no longer load a local PDF on startup. The API is ready to accept URLs.
    logger.info("API is ready to receive requests with a 'documents' URL in the body.")

# --- API ENDPOINTS ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Documents from a URL"
)
async def run_submission(request: Request):
    """Processes a list of questions against a dynamically loaded PDF document from a URL and any URLs found within it."""
    
    # We no longer use `default_vector_store`
    vector_store: Optional[FAISS] = None

    try:
        raw_body = request.state.body
        if not raw_body:
            raise ValueError("Request body is empty.")
        request_body = QueryRequest.parse_raw(raw_body)
        
        if not request_body.documents:
            raise ValueError("A 'documents' URL is required in the request body.")
            
        pdf_url = request_body.documents
        
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request body format or missing 'documents' URL. Error: {e}")

    logger.info(f"Processing request for {len(request_body.questions)} questions with PDF URL: {pdf_url}")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed.")

    # --- NEW DYNAMIC DOCUMENT LOADING LOGIC ---
    try:
        # Step 1: Download the PDF from the provided URL
        logger.info(f"Downloading PDF from {pdf_url}...")
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content
        
        # Step 2: Load the PDF content into PyPDFLoader from memory
        loader = PyPDFLoader(io.BytesIO(pdf_content))
        documents = loader.load()
        
        # All documents from the PDF will be stored here
        all_documents = documents.copy()
        
        # Step 3: Extract URLs from the PDF content
        pdf_text = " ".join([doc.page_content for doc in documents])
        puzzle_urls = extract_urls_from_string(pdf_text)
        
        # Step 4: Fetch content from each puzzle URL
        for url in set(puzzle_urls): # Use a set to avoid processing duplicate URLs
            try:
                # Use the Browse tool to fetch the content from the embedded URL
                logger.info(f"Fetching content from puzzle URL: {url}...")
                content = await Browse(url=url, query="full content")
                if content:
                    all_documents.append(Document(page_content=content, metadata={"source": url}))
            except Exception as e:
                logger.warning(f"Could not fetch content from embedded URL {url}. Error: {e}")

        logger.info(f"Loaded {len(documents)} pages from the main PDF and {len(all_documents) - len(documents)} additional web pages.")

        # Step 5: Split the combined documents and create the vector store
        logger.info("Splitting combined documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(docs)} text chunks for RAG processing.")
        
        logger.info("Creating embeddings and building FAISS vector store...")
        vector_store = await embed_with_retries(docs)
        logger.info("FAISS vector store built successfully for this request.")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error downloading PDF from {pdf_url}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Failed to download PDF from the provided URL. Error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during document processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during document processing: {e}")

    answers = []
    for question in request_body.questions:
        try:
            answer = await process_question_with_retries(question, vector_store)
            answers.append(answer)
        except httpx.HTTPStatusError as e:
            logger.error(f"Final failure after all retries for question '{question}': {e}")
            answers.append(f"Could not retrieve an answer due to API quota limits or network issues. Error: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing question '{question}': {e}")
            answers.append(f"An unexpected internal error occurred: {e}")
    
    # Introduce a single, longer delay of 10 seconds after processing all questions
    logger.info("All questions processed. Waiting for 10 seconds before sending the final response to cool down the API.")
    await asyncio.sleep(1)
    
    logger.info("--- Sending response. ---")
    return {"answers": answers}

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """A simple health check endpoint."""
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
