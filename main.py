import os
import requests
from dotenv import load_dotenv
import uuid
from pathlib import Path

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

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration & Setup ---

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your .env file or Render environment.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please add it to your .env file or Render environment.")

# Define the path to the merged PDF document (fallback/initial document)
PDF_PATH = "policy.pdf"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Google Gemini)",
    description="API for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Global RAG Chain Components ---
default_vector_store: Optional[FAISS] = None
default_qa_chain: Optional[RetrievalQA] = None


# --- Prompt Engineering with Few-Shot Examples ---
# These examples directly teach the LLM the desired output format and style.
# The citation format is used within the example answers to simulate document referencing.
# The actual source document page numbers are manually added for demonstration.
# IMPORTANT: The actual citations will depend on the RAG system's ability to extract metadata,
# which the current setup (return_source_documents=False for simplicity) doesn't directly expose
# to the LLM for generation in the string answer. For true dynamic citations, a more complex
# chain that processes source_documents from RetrievalQA.invoke is needed.
# For this prompt, the citations are illustrative of the desired *style*.
PROMPT_TEMPLATE = """
You are an expert in analyzing policy documents, contracts, and emails.
Your task is to answer user queries accurately and concisely, based **only** on the provided context.
If the answer is not found in the context, state: "I cannot answer this question based on the provided documents."
Do not generate information that is not supported by the context.
When providing an answer, aim for directness and precision, summarizing the key information from the policy.

BEGIN EXAMPLES:
Example Question: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
[cite_start]Example Answer: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits. [cite: 44]

Example Question: What is the waiting period for pre-existing diseases (PED) to be covered?
[cite_start]Example Answer: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered. [cite: 369]

Example Question: Does this policy cover maternity expenses, and what are the conditions?
Example Answer: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.

Example Question: What is the waiting period for cataract surgery?
[cite_start]Example Answer: The policy has a specific waiting period of two (2) years for cataract surgery. [cite: 371, 387]

Example Question: Are the medical expenses for an organ donor covered under this policy?
[cite_start]Example Answer: Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994. [cite: 201, 202]

Example Question: What is the No Claim Discount (NCD) offered in this policy?
Example Answer: A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. [cite_start]The maximum aggregate NCD is capped at 5% of the total base premium. [cite: 637, 638, 639]

Example Question: Is there a benefit for preventive health check-ups?
Example Answer: Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. [cite_start]The amount is subject to the limits specified in the Table of Benefits. [cite: 204]

Example Question: How does the policy define a 'Hospital'?
[cite_start]Example Answer: A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients. [cite: 45, 46, 47, 48]

Example Question: What is the extent of coverage for AYUSH treatments?
[cite_start]Example Answer: The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital. [cite: 207, 407]

Example Question: Are there any sub-limits on room rent and ICU charges for Plan A?
Example Answer: For Domestic Cover, Room rent and Boarding expenses are covered without any sub-limit. [cite_start]ICU expenses are paid up to actual ICU expenses provided by the Hospital. [cite: 184, 185]

Example Question: What is the age limit for an Insured Person?
[cite_start]Example Answer: An Insured Person must be between 3 months and 65 years of age at the commencement of the first Global Health Care Policy. [cite: 142]

Example Question: How many days of pre-hospitalization medical expenses are covered?
[cite_start]Example Answer: Medical Expenses incurred during the 60 days immediately before hospitalization are covered, provided they are for the same condition for which subsequent hospitalization was required and the inpatient claim is accepted. [cite: 190]

Example Question: How many days of post-hospitalization medical expenses are covered?
[cite_start]Example Answer: Medical expenses incurred during the 180 days immediately after discharge from hospitalization are covered, provided they are for the same condition for which earlier hospitalization was required and the inpatient claim is accepted. [cite: 196]

Example Question: What is the maximum reimbursement for local road ambulance?
Example Answer: The policy will pay the reasonable cost, specified in the Policy Schedule, for local road ambulance. [cite_start]Claims are payable only if the life-threatening emergency condition is certified by a Medical Practitioner and an Inpatient Hospitalization or Day Care Procedures claim is accepted. [cite: 197, 198, 199]

Example Question: Does the policy cover Mental Illness Treatment, and what are its exclusions?
Example Answer: Yes, the policy covers Customary and Reasonable expenses for In-patient treatment of Mental Illness (as specified under Annexure IV) in a recognized psychiatric unit of a Hospital, up to the Sum Insured. [cite_start]Exclusions include expenses related to Alcoholism, drug or substance abuse, diagnostic tests without psychiatrist advice, alternate treatments other than Allopathic, autism spectrum disorder admissions at specialized educational facilities, and Out-patient Treatment. [cite: 218, 222, 223, 224]

Example Question: What is the definition of a Pre-Existing Disease (PED)?
[cite_start]Example Answer: A Pre-Existing Disease is any condition, ailment, injury or disease diagnosed by a physician or for which medical advice/treatment was recommended/received within 48 months prior to the effective date of the policy or its reinstatement. [cite: 87, 88]

Example Question: What is a 'Day Care Treatment'?
Example Answer: Day care treatment means medical and/or surgical procedures undertaken under General or Local Anesthesia in a Hospital/Day Care Centre in less than 24 hours due to technological advancement, which would otherwise require over 24 hours of hospitalization. [cite_start]Out-patient basis treatment is excluded. [cite: 33, 34, 35]

Example Question: Is there a co-payment for Dental Plan Benefits for international cover?
[cite_start]Example Answer: Yes, there is a mandatory Co-Payment of 20% on each and every claim under Dental Plan Benefits for international cover. [cite: 347]

Example Question: What is the claim settlement period for domestic cover?
Example Answer: The Company shall settle or reject a claim within 30 days from the date of receipt of the last necessary document. [cite_start]In cases requiring investigation, the period extends to 45 days. [cite: 553, 554]

Example Question: What happens if there's a delay in claim payment for domestic cover?
[cite_start]Example Answer: In case of delay in payment of a claim, the Company is liable to pay interest to the Policyholder at a rate 2% above the bank rate from the date of receipt of the last necessary document to the date of claim payment. [cite: 554, 555]

Example Question: What is the policy on 'Multiple Policies'?
Example Answer: If an Insured has multiple policies from the same or different insurers, they have the right to choose which policy to claim from. [cite_start]If the Sum Insured of a single policy is exhausted, they can claim the balance from another policy, subject to its terms and conditions. [cite: 557, 558]

Example Question: Under what conditions can the company cancel the policy?
Example Answer: The Company may cancel the policy at any time on grounds of misrepresentation, non-disclosure of material facts, or fraud by the insured person, by giving 15 days' written notice. [cite_start]In such cases, there would be no refund of premium. [cite: 572]

Example Question: What is the maximum sum insured for Air Ambulance under Imperial Plan?
[cite_start]Example Answer: For the Imperial Plan, Air Ambulance expenses are reimbursed up to INR 500,000. [cite: 239]

Example Question: Are diagnostic tests covered under Out-patient Treatment for International Cover (Imperial Plus Plan)?
[cite_start]Example Answer: Yes, Diagnostic tests are covered under Out-patient Treatment for International Cover (Imperial Plus Plan) up to the limits specified in the Policy Schedule. [cite: 337]

Example Question: What defines a "Network Provider"?
[cite_start]Example Answer: A Network Provider means Hospitals or healthcare providers enlisted by the insurer, TPA, or jointly by an Insurer and TPA to provide medical services to an Insured by a Cashless Facility. [cite: 77]

Example Question: What is the maximum percentage of sum insured for ICU expenses under Hospitalization for Domestic Cover?
Example Answer: Intensive Care Unit (ICU) expenses are covered up to 5% of the Sum Insured, subject to a maximum of Rs. [cite_start]10,000 per day. [cite: 1697]

Example Question: Are Dental Treatment and Surgery covered under Domestic Cover?
Example Answer: Dental treatment is covered if necessitated due to disease or injury. [cite_start]Dental cosmetic surgery, dentures, dental prosthesis, dental implants, orthodontics, surgery of any kind are excluded unless as a result of Accidental Bodily Injury to natural teeth and requiring Hospitalization. [cite: 419, 431]

Example Question: What are the primary details that should be in a medical practitioner's prescription?
[cite_start]Example Answer: A medical practitioner's prescription should name the Insured Person and, for drugs, specify the drugs prescribed, their price, and include a receipt for payment. [cite: 1649]

Example Question: What is the definition of "Accident"?
[cite_start]Example Answer: An Accident means a sudden, unforeseen and involuntary event caused by external, visible and violent means. [cite: 7]

Example Question: What is "Any one Illness"?
[cite_start]Example Answer: Any one Illness means a continuous Period of Illness and it includes relapse within 45 days from the date of last consultation with the Hospital/Nursing Home where treatment was taken. [cite: 8]

Example Question: How much is the daily allowance for choosing shared accommodation under HDFC ERGO Easy Health Individual Exclusive Plan for Rs. 500000 Sum Insured?
Example Answer: For the Easy Health Individual Exclusive Plan with a Sum Insured of Rs. 500,000, the daily cash for choosing shared accommodation is Rs. 800 per day, with a maximum of Rs. [cite_start]4,800. [cite: 1655]

Example Question: Does the Cholamandalam MS Group Domestic Travel Insurance cover any pre-existing conditions?
Example Answer: This policy is not designed to provide an indemnity with respect to medical services the need for which arises out of a pre-existing condition as defined in the policy in normal course of treatment. [cite_start]However in any of the threatening situation this exclusion shall not be applied and also that the cover will up to the limit shown under Life threatening condition/ situation as defined in this policy. [cite: 1036]

Example Question: What is the entry age for members under the Cholamandalam MS Group Domestic Travel Insurance?
[cite_start]Example Answer: Entry age for the member should be between 03 months to 90 years (completed age). [cite: 967]

Example Question: Is physiotherapy covered under HDFC ERGO Easy Health Domestic Plan?
[cite_start]Example Answer: Yes, physiotherapy is covered under Pre-Hospitalization Medical Expenses and Post-Hospitalization Medical Expenses if prescribed by a Medical Practitioner and is Medically Necessary Treatment. [cite: 1630]

Example Question: What is the grace period for renewal of National Arogya Sanjeevani Policy?
Example Answer: The Grace Period for payment of the premium shall be thirty days. [cite_start]In case of Renewal, Coverage shall not be available during the period for which no premium is received. [cite: 1694]

Example Question: Are spectacles and contact lenses covered under HDFC ERGO Easy Health policy?
[cite_start]Example Answer: No, the provision or fitting of hearing aids, spectacles or contact lenses including optometric therapy are excluded. [cite: 1640]

Example Question: What is the maximum liability for Air Ambulance under Edelweiss Well Baby Well Mother add-on?
Example Answer: The maximum liability under this benefit for any and all claims arising during the Policy Year will be restricted to the Sum insured as stated in the Policy Schedule. The maximum distance of travel undertaken is 150 kms. [cite_start]In case of distance travelled is more than 150 kms, proportionate amount of expenses upto 150 kms shall be payable. [cite: 1620]

Example Question: Is medical error covered under Bajaj Allianz Global Health Care policy?
[cite_start]Example Answer: No, treatment required as a result of medical error is excluded. [cite: 440]

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
    global default_qa_chain, default_vector_store

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        print("Default FAISS vector store built successfully.")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0) # Lower temperature for factual answers
        
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(search_kwargs={"k": 5}), # Using k=5, can be tuned
            chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Apply custom prompt here
        )
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your GOOGLE_API_KEY is correct, 'policy.pdf' exists, and all required packages (like faiss-cpu) are installed.")


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
    current_vector_store = None
    current_qa_chain = None
    temp_doc_path = None

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")

    try:
        # Determine which document source to use
        if request_body.documents:
            print(f"Processing request with dynamic document URL: {request_body.documents}")
            documents, temp_doc_path = await _fetch_and_load_document_from_url(request_body.documents)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            print(f"Created {len(docs)} text chunks for dynamic document.")

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            current_vector_store = FAISS.from_documents(docs, embeddings)
            print("FAISS vector store built for dynamic document.")

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
            
            current_qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=current_vector_store.as_retriever(search_kwargs={"k": 5}), # Using k=5
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
            current_qa_chain = default_qa_chain

        answers = []
        for question in request_body.questions:
            print(f"Processing question: '{question}'")
            try:
                result = current_qa_chain.invoke({"query": question})
                answers.append(result.get("result", "Could not retrieve an answer based on the provided documents."))
            except Exception as qa_e:
                print(f"ERROR: Failed to process question '{question}' with RAG chain: {qa_e}")
                answers.append(f"An error occurred while processing this question: {qa_e}")
            print(f"Answer generated for '{question}'.")

        print("--- All questions processed. Sending response. ---")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during overall query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        if temp_doc_path and temp_doc_path.exists():
            try:
                os.remove(temp_doc_path)
                print(f"Cleaned up temporary file: {temp_doc_path}")
                if Path("./temp_docs").exists() and not any(Path("./temp_docs").iterdir()):
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
