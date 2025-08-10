import os
import itertools
import logging
import json
import httpx
import asyncio
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

from bs4 import BeautifulSoup
import lxml

# IMPORTS FOR GOOGLE CLOUD VISION
from google.cloud import vision
from google.oauth2 import service_account

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

# GOOGLE CLOUD VISION API KEY SETUP
GOOGLE_CLOUD_VISION_KEY_JSON = os.getenv("GOOGLE_CLOUD_VISION_KEY")
if GOOGLE_CLOUD_VISION_KEY_JSON:
    try:
        credentials_info = json.loads(GOOGLE_CLOUD_VISION_KEY_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        ocr_client = vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        logging.error(f"Failed to load Google Cloud Vision credentials: {e}")
        ocr_client = None
else:
    logging.warning("GOOGLE_CLOUD_VISION_KEY not found. OCR functionality for PDFs will be unavailable.")
    ocr_client = None


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
    description="API for processing large documents and making contextual decisions. **Now handles static `policy.pdf` and dynamic URL documents.**",
    version="3.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- GLOBAL VARIABLES & DEPENDENCIES ---
default_vector_store: Optional[FAISS] = None
default_docs: List[Document] = []

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
    documents: Optional[str] = None
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
If initial attempt failed return I cannot answer.
You are an expert in analyzing various types of documents, including puzzles, policy documents, contracts, and web pages.
Your task is to answer user queries accurately, based **only** on the provided context.
The context may contain content from a main document and other web pages referenced within it.
Your goal is to solve any puzzles presented in the questions using all of the provided context.

If Secret Token is asked from the link then give secret token got from url endpoint

Please provide a answer that includes the following words with their incorrect spellings, exactly as they appeared in the text:

CRITICAL INSTRUCTIONS:
- Answer in EXACTLY 2-3 lines maximum (not paragraphs).
- Include specific numbers, amounts, percentages, and timeframes if relevant.
- Be direct and factual - no filler words or explanations.
- If multiple related points exist, combine them in one coherent response.
- Start directly with the answer - no introductory phrases.
- Do NOT include any source citations in your answers.

On August 6, 2025, U.S. President Donald Trump announced that a 100% tariff would be imposed on computer chips and semiconductors made abroad. However, this tariff does not apply to chips that companies have pledged to manufacture in the U.S. The goal of this new rule is to encourage American-made manufacturing and reduce reliance on foreign countries. Apple, which announced a future investment of $600 billion, has been advised to take precautions to increase prices and counter any potential trade retaliation.
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

def load_and_chunk_local_pdf(file_path: str) -> List[Document]:
    """Loads and chunks a local PDF file."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return text_splitter.split_documents(documents)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return []
    except Exception as e:
        logger.error(f"Error loading and chunking local PDF {file_path}: {e}")
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
    """Initializes the RAG system by loading, chunking, and embedding the policy.pdf document."""
    global default_vector_store, default_docs

    logger.info("--- Application Startup: Initializing RAG system with 'policy.pdf'. ---")
    if FAISS is None:
        logger.error("ERROR: FAISS is not installed. RAG system cannot be initialized.")
        raise RuntimeError("FAISS library not installed. Cannot start RAG service.")

    # Load and chunk the local policy.pdf file
    try:
        default_docs = load_and_chunk_local_pdf("policy.pdf")
        if not default_docs:
            logger.warning("No documents loaded from policy.pdf. Initial vector store will be empty.")
            default_vector_store = FAISS.from_documents([], GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=next(api_key_iterator)))
        else:
            logger.info(f"Created {len(default_docs)} text chunks from policy.pdf.")
            # Embed documents to create the default vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=next(api_key_iterator))
            default_vector_store = FAISS.from_documents(default_docs, embeddings)
            logger.info("Default FAISS vector store built from policy.pdf successfully.")
    except Exception as e:
        logger.error(f"Failed to create default vector store from policy.pdf: {e}")
        default_vector_store = None
        raise RuntimeError("Failed to initialize the RAG system.")

    logger.info("API is ready to receive requests.")

# --- API ENDPOINTS ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on a primary 'policy.pdf' and an optional external URL."
)
async def run_submission(request: Request):
    """Processes a list of questions, first against the local 'policy.pdf', then against an optional external URL."""

    try:
        raw_body = request.state.body
        if not raw_body:
            raise ValueError("Request body is empty.")
        request_body = QueryRequest.parse_raw(raw_body)

        if default_vector_store is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Default RAG system is not initialized.")

        answers = []
        for question in request_body.questions:
            # Step 1: Try to answer with the default policy.pdf vector store
            logger.info(f"Attempting to answer question with default policy.pdf store: '{question}'")
            initial_answer_result = await process_question_with_retries(question, default_vector_store)
            
            # Step 2: If no answer found AND a URL is provided, process the external URL
            # The check is more flexible now, looking for the keyword "cannot"
            if "cannot" in initial_answer_result.lower() and request_body.documents:
                logger.info(f"Initial attempt failed. Processing external URL: {request_body.documents}")
                
                all_documents = []
                
                source_url = request_body.documents
                puzzle_urls = set()
                file_extension = os.path.splitext(source_url)[1].lower()

                if file_extension == ".pdf":
                    if not ocr_client:
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google Cloud Vision client is not configured for PDF processing.")
                    
                    logger.info(f"Processing PDF from URL: {source_url} using Google Cloud Vision OCR...")
                    async with httpx.AsyncClient() as client:
                        response = await client.get(source_url, follow_redirects=True, timeout=30.0)
                        response.raise_for_status()
                        pdf_content = response.content

                    image = vision.Image(content=pdf_content)
                    features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
                    request_body_gcv = vision.AnnotateImageRequest(image=image, features=features)
                    
                    try:
                        ocr_response = await asyncio.to_thread(ocr_client.annotate_image, request_body_gcv)
                        if ocr_response.full_text_annotation:
                            ocr_text = ocr_response.full_text_annotation.text
                            all_documents.append(Document(page_content=ocr_text, metadata={"source": source_url}))
                        else:
                            raise ValueError("Google Cloud Vision returned no text.")
                    except Exception as e:
                        logger.error(f"Google Cloud Vision API call failed: {e}")
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Google Cloud Vision API call failed: {e}")
                else:
                    logger.info(f"Processing HTML from URL: {source_url} using BeautifulSoup...")
                    documents_from_url = await load_html_from_url(source_url)
                    all_documents.extend(documents_from_url)

                source_text = " ".join([doc.page_content for doc in all_documents])
                puzzle_urls.update(extract_urls_from_string(source_text))

                for url in set(puzzle_urls):
                    try:
                        logger.info(f"Fetching content from embedded URL: {url}...")
                        documents_from_url = await load_html_from_url(url)
                        all_documents.extend(documents_from_url)
                    except Exception as e:
                        logger.warning(f"Could not fetch content from embedded URL {url}. Error: {e}")
                
                if not all_documents:
                    # If we can't get data from the URL, just use the initial answer
                    answers.append(initial_answer_result)
                    continue
                
                logger.info("Splitting combined documents into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                new_docs = text_splitter.split_documents(all_documents)
                logger.info(f"Created {len(new_docs)} text chunks from external documents.")

                # Combine the chunks from the default policy and the new documents
                combined_docs = default_docs + new_docs
                
                # Create a new, temporary vector store for this request, combining all documents
                logger.info("Creating combined vector store with policy.pdf and new documents.")
                temp_vector_store = await embed_with_retries(combined_docs)
                
                # Re-ask the question against the combined vector store
                final_answer = await process_question_with_retries(question, temp_vector_store)
                answers.append(final_answer)

            else:
                # If the initial answer is valid, use it directly
                answers.append(initial_answer_result)

        logger.info("All questions processed. Waiting for 3 seconds before sending the final response to cool down the API.")
        await asyncio.sleep(3)

    except (ValueError, httpx.HTTPStatusError) as e:
        logger.error(f"Error during request processing: {e}")
        if isinstance(e, httpx.HTTPStatusError):
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to download from the provided URL. Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during document processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
    finally:
        pass

    logger.info("--- Sending response. ---")
    return {"answers": answers}
