import os
import itertools
import logging
import json
import httpx
import asyncio
import re
import io
import tempfile
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# New loader for PDFs that is more robust with text extraction
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from bs4 import BeautifulSoup
import lxml

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
    description="API for processing large documents and making contextual decisions. **Now handles dynamic PDF and HTML URLs.**",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- GLOBAL VARIABLES & DEPENDENCIES ---
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
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
You are an expert in analyzing various types of documents, including puzzles, policy documents, contracts, and web pages.
Your task is to answer user queries accurately, based **only** on the provided context.
The context may contain content from a main document and other web pages referenced within it.
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

def extract_urls_from_string(text: str) -> List[str]:
    """Finds and returns all valid URLs in a given string."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)

async def load_html_from_url(url: str) -> List[Document]:
    """Loads and parses HTML from a URL using BeautifulSoup."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            text = soup.get_text(separator=' ', strip=True)
            return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        logger.error(f"Failed to load HTML from {url}: {e}")
        return []

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
    
    logger.info("API is ready to receive requests with a 'documents' URL in the body.")

# --- API ENDPOINTS ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Documents from a URL"
)
async def run_submission(request: Request):
    """Processes a list of questions against a dynamically loaded document from a URL and any URLs found within it."""
    
    vector_store: Optional[FAISS] = None
    temp_file_path: Optional[str] = None

    try:
        raw_body = request.state.body
        if not raw_body:
            raise ValueError("Request body is empty.")
        request_body = QueryRequest.parse_raw(raw_body)
        
        if not request_body.documents:
            raise ValueError("A 'documents' URL is required in the request body.")
            
        source_url = request_body.documents
        
        if FAISS is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed.")

        all_documents = []
        
        file_extension = os.path.splitext(source_url)[1].lower()

        if file_extension == ".pdf":
            logger.info(f"Processing PDF from URL: {source_url}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = temp_file.name
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(source_url, follow_redirects=True, timeout=30.0)
                    response.raise_for_status()
                    temp_file.write(response.content)

                # Use the PyPDFium2Loader for its robust text extraction
                loader = PyPDFium2Loader(temp_file_path)
                documents = loader.load()
                all_documents.extend(documents)
        else:
            logger.info(f"Processing HTML from URL: {source_url} using BeautifulSoup...")
            documents = await load_html_from_url(source_url)
            all_documents.extend(documents)
        
        source_text = " ".join([doc.page_content for doc in all_documents])
        puzzle_urls = extract_urls_from_string(source_text)
        
        for url in set(puzzle_urls):
            try:
                logger.info(f"Fetching content from puzzle URL: {url}...")
                documents_from_url = await load_html_from_url(url)
                all_documents.extend(documents_from_url)
            except Exception as e:
                logger.warning(f"Could not fetch content from embedded URL {url}. Error: {e}")

        logger.info(f"Loaded a total of {len(all_documents)} documents from the main URL and embedded links.")
        
        logger.info("Splitting combined documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(docs)} text chunks for RAG processing.")
        
        logger.info("Creating embeddings and building FAISS vector store...")
        vector_store = await embed_with_retries(docs)
        logger.info("FAISS vector store built successfully for this request.")

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
    
        logger.info("All questions processed. Waiting for 10 seconds before sending the final response to cool down the API.")
        await asyncio.sleep(10)

    except (ValueError, httpx.HTTPStatusError) as e:
        logger.error(f"Error during request processing: {e}")
        if isinstance(e, httpx.HTTPStatusError):
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to download from the provided URL. Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during document processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Deleted temporary file: {temp_file_path}")

    logger.info("--- Sending response. ---")
    return {"answers": answers}

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """A simple health check endpoint."""
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
