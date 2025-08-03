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

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict

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

load_dotenv()

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
    global current_mistral_api_key
    current_mistral_api_key = next(api_key_iterator)
    logger.info(f"Switched to next Mistral API Key. Current key (partial): {current_mistral_api_key[:5]}...")
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

# Check if handlers already exist to prevent adding multiple times on reload
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    # 'documents' field is kept for API compatibility but will be ignored for processing.
    documents: Optional[str] = None
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Middleware to log incoming requests ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log the incoming request method and URL
    logger.info(f"Incoming Request: {request.method} {request.url}")

    # For /hackrx/run, attempt to log the request body as well
    if request.url.path == "/hackrx/run" and request.method == "POST":
        try:
            # Read the request body. This consumes the stream.
            body = await request.body()
            # Store it in request.state so the endpoint can access it later
            request.state.body = body
            
            # Attempt to decode as JSON for logging purposes
            req_data = json.loads(body.decode('utf-8'))
            # Log the questions array from the request body
            logger.info(f"Request Questions: {req_data.get('questions', 'N/A')}")
            # Explicitly ignore 'documents' field for processing
            if 'documents' in req_data and req_data['documents']:
                logger.info("NOTE: 'documents' URL received in request body but will be ignored for processing.")
        except json.JSONDecodeError:
            logger.warning("Could not decode request body as JSON.")
            request.state.body = b'' # Ensure it's bytes even if parsing fails
        except Exception as e:
            logger.exception(f"Unexpected error reading/parsing request body in middleware: {e}")
            request.state.body = b''
    else:
        # For non-POST /hackrx/run requests, no body to read/store
        pass

    response = await call_next(request)
    return response

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

    logger.info("--- Application Startup: Initializing RAG System with default policy.pdf ---")

    if not os.path.exists(PDF_PATH):
        logger.error(f"ERROR: Default '{PDF_PATH}' not found. The API cannot function without this document as dynamic URLs are ignored.")
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
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=current_mistral_api_key)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        logger.info("Default FAISS vector store built successfully.")

        llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
        
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
        default_chain_initialized_with_key = current_mistral_api_key
        logger.info("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        logger.exception(f"--- ERROR during Default RAG System Initialization: {e} ---")
        logger.error("Please ensure your MISTRAL_API_KEYS are correct, 'policy.pdf' exists, and all required packages (like faiss-cpu, langchain-mistralai) are installed.")
        raise RuntimeError(f"RAG system initialization failed: {e}")

# --- API Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse, # This line was the reported error line
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Query-Retrieval on Policy Documents"
)
async def run_submission(request: Request):
    global default_qa_chain, default_vector_store, current_mistral_api_key, default_chain_initialized_with_key

    # Reconstruct request_body from the state set by the middleware
    try:
        # Access the body stored by the middleware
        if not hasattr(request.state, 'body') or not request.state.body:
             # This branch should ideally not be hit if middleware is working, but provides a fallback
             raw_body = await request.body() # Read directly if not in state
        else:
            raw_body = request.state.body
            
        request_body = QueryRequest.parse_raw(raw_body)
        
        # Explicitly ignore documents field, even if present in the Pydantic model.
        # This is already handled by the model definition, but reinforcing intent.
        if request_body.documents is not None:
            logger.warning("The 'documents' field in the request body is present but will be ignored for processing as per system design.")

    except Exception as e:
        logger.error(f"Failed to parse request body in run_submission or access state: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request body format.")

    logger.info(f"Processing request for {len(request_body.questions)} questions.")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")
    
    if default_qa_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. 'policy.pdf' might be missing or there were startup errors. Check server logs."
        )

    answers = []
    
    max_retries_per_question = len(MISTRAL_API_KEYS) * 2
    for question in request_body.questions:
        logger.info(f"Processing question: '{question}'")
        attempt = 0
        answer_found = False
        while attempt < max_retries_per_question:
            try:
                # Log when a key rotation leads to chain re-initialization
                if default_chain_initialized_with_key != current_mistral_api_key:
                    logger.info(f"API Key Rotation: Re-initializing default chain with new key (partial): {current_mistral_api_key[:5]}...")
                    llm_for_default = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=current_mistral_api_key)
                    
                    default_qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_for_default,
                        chain_type="stuff",
                        retriever=default_vector_store.as_retriever(search_kwargs={"k": 4}),
                        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                    )
                    default_chain_initialized_with_key = current_mistral_api_key

                result = default_qa_chain.invoke({"query": question})
                answers.append(result.get("result", "I cannot answer this question based on the provided documents."))
                answer_found = True
                break

            except (MistralAPIException, requests.exceptions.RequestException) as e:
                attempt += 1
                logger.warning(f"API error for current key (attempt {attempt}/{max_retries_per_question}) for question '{question}': {e}")
                if attempt < max_retries_per_question:
                    get_next_api_key()
                    logger.info("Retrying with new key after a short delay...")
                    time.sleep(5)
                else:
                    logger.error("All API keys exhausted or max retries reached for question. Failing this question.")
                    answers.append(f"Could not retrieve an answer due to API quota limits being exceeded or persistent network issues.")
                    break
            except Exception as e:
                logger.exception(f"ERROR: Failed to process question '{question}' with RAG chain (unexpected error): {e}")
                answers.append(f"An unexpected error occurred: {e}")
                answer_found = True
                break
            
            if not answer_found and attempt >= max_retries_per_question:
                answers.append(f"Failed to answer '{question}' after multiple retries due to persistent API quota limits or other issues.")

        logger.info(f"Answer generated for '{question}'.")

    logger.info("--- All questions processed. Sending response. ---")
    return {"answers": answers}

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "LLM-Powered Intelligent Query–Retrieval System API is running. Visit /api/v1/docs for interactive documentation."}
