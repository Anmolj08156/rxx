import os
import requests
from dotenv import load_dotenv
import uuid
from pathlib import Path
import json # For parsing LLM output
import re # For regex to clean LLM output

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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

PDF_PATH = "policy.pdf" # This PDF is expected to be in the root of the project directory

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Claim Processing System (Google Gemini)",
    description="API for processing insurance claims based on policy documents, providing structured decisions and justifications.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Global RAG Chain Components ---
default_vector_store: Optional[FAISS] = None
default_qa_chain: Optional[RetrievalQA] = None
llm_parser_model: Optional[ChatGoogleGenerativeAI] = None # Dedicated LLM for parsing claims


# --- Prompt Engineering ---

# Prompt for the main claim processing LLM
INSURANCE_CLAIM_PROMPT = """
You are an expert insurance claim processor with deep knowledge of policy terms, coverage rules, and claim evaluation. You must analyze claims systematically and provide structured decisions.
Your primary goal is to answer the user's "Claim Question" based **strictly** on the "Policy Context" and "Claim Details".

ANALYSIS FRAMEWORK:
1. **Eligibility Assessment**: Determine if the claim is covered under the policy based on provided claim details.
2. **Coverage Limits**: Identify applicable limits, deductibles, and caps.
3. **Coordination of Benefits**: Check for multiple insurance policies and calculate remaining amounts if applicable.
4. **Exclusion Review**: Identify any policy exclusions that apply based on the claim.
5. **Decision Logic**: Apply business rules to determine approval/denial based on policy context and claim details.
6. **Payout Calculation**: Calculate exact amounts considering all factors (limits, deductibles, primary payments).

RESPONSE FORMAT (MUST BE VALID JSON - ensure all fields are present or null/empty array as appropriate):
{{
    "decision": "[APPROVED/DENIED/PENDING_REVIEW]",
    "confidence_score": [0.0-1.0],
    "payout_amount": [amount or null],
    "reasoning": "A concise and direct answer to the Claim Question, summarizing the key information from the policy and incorporating Claim Details.",
    "policy_sections_referenced": ["section_name_or_clause_number (Document: filename.pdf, Page X)", "another_section (Document: filename.docx, Page Y)"],
    "exclusions_applied": ["exclusion_clause_or_name (Document: filename.pdf, Page Z)", "another_exclusion"],
    "coordination_of_benefits": {{
        "has_other_insurance": [true/false],
        "primary_insurance": "name or null",
        "secondary_insurance": "name or null",
        "primary_payment": [amount or null],
        "remaining_amount": [amount or null]
    }},
    "processing_notes": ["note1", "note2"]
}}

IMPORTANT RULES:
- Base decisions ONLY on information in the "Policy Context" and "Claim Details".
- For coordination of benefits, if 'has_other_insurance' is true, assume 'primary_payment' is the amount already paid by primary insurer. Calculate 'remaining_amount' from the 'requested_amount' and policy limits, considering the 'primary_payment'. If 'requested_amount' is not provided, 'payout_amount' and 'remaining_amount' should be null.
- Include 'confidence_score' based on clarity of policy language and completeness of claim details.
- **VERY IMPORTANT: For 'policy_sections_referenced' and 'exclusions_applied', you MUST reference specific policy sections (e.g., 'Section A', 'Clause 3.1') AND the original document/page information (e.g., 'Document: policy.pdf, Page 3') that was present in the retrieved context.** Extract these details directly from the source document metadata if possible.
- If information is unclear, ambiguous, or missing for a conclusive decision, use "PENDING_REVIEW" decision and explain why in 'reasoning' and 'processing_notes'.
- The 'reasoning' field MUST contain the direct answer to the 'Claim Question', formatted clearly and concisely.
- Ensure the output is a single, perfectly valid JSON object. Do not include any text before or after the JSON.

Policy Context:
{context}

Claim Details (Structured, if available):
{claim_details_json}

Claim Question: {question}

Insurance Analysis (JSON format only):
"""
CUSTOM_CLAIM_PROMPT = PromptTemplate(template=INSURANCE_CLAIM_PROMPT, input_variables=["context", "claim_details_json", "question"])


# Prompt for LLM Parser to extract structured details from natural language query
LLM_PARSER_PROMPT = """
You are an AI assistant designed to extract key claim details from natural language queries.
Your goal is to parse the user's question and identify structured information relevant to an insurance claim.

Extract the following details if mentioned or inferable:
- 'patient_age' (integer)
- 'procedure' (string)
- 'location' (string)
- 'policy_duration_months' (integer, infer from phrases like '3-month-old policy', 'policy for X months')
- 'requested_amount' (float, from phrases like '$X', 'Y USD', 'Z rupees')
- 'has_other_insurance' (boolean: true if phrases like 'other insurance', 'secondary claim', 'already paid by primary' are present, else false)
- 'primary_insurance_payment' (float, if a specific amount paid by primary insurer is mentioned, e.g., '$X paid by primary')

Return the extracted details in JSON format. If a detail is not found or cannot be inferred, omit it from the JSON.
Ensure the output is a single, perfectly valid JSON object. Do not include any text before or after the JSON.

Example Input: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy, seeking $25,000, with $10,000 paid by primary insurance"
Example Output:
{{
  "patient_age": 46,
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration_months": 3,
  "requested_amount": 25000.0,
  "has_other_insurance": true,
  "primary_insurance_payment": 10000.0
}}

Query: {query}

Extracted Claim Details (JSON format only):
"""
CUSTOM_PARSER_PROMPT = PromptTemplate(template=LLM_PARSER_PROMPT, input_variables=["query"])


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

# Nested models for the structured JSON response
class CoordinationOfBenefits(BaseModel):
    has_other_insurance: bool = Field(False, description="True if other insurance policies are involved.")
    primary_insurance: Optional[str] = Field(None, description="Name of the primary insurer, if applicable.")
    secondary_insurance: Optional[str] = Field(None, description="Name of the secondary insurer, if applicable.")
    primary_payment: Optional[float] = Field(None, description="Amount paid by the primary insurer, if applicable.")
    remaining_amount: Optional[float] = Field(None, description="Remaining amount after primary payment and policy limits.")

class ClaimResponseDetail(BaseModel):
    decision: str = Field(..., description="Decision on the claim (APPROVED/DENIED/PENDING_REVIEW).")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision (0.0 to 1.0).")
    payout_amount: Optional[float] = Field(None, description="Calculated payout amount, if applicable.")
    reasoning: str = Field(..., description="Concise justification for the decision, including key policy information.")
    policy_sections_referenced: List[str] = Field([], description="List of specific policy sections/pages referenced.")
    exclusions_applied: List[str] = Field([], description="List of exclusions applied, if any.")
    coordination_of_benefits: CoordinationOfBenefits = Field(
        default_factory=CoordinationOfBenefits, description="Details regarding coordination of benefits."
    )
    processing_notes: List[str] = Field([], description="Any additional notes during processing.")

class QueryRequest(BaseModel):
    documents: Optional[str] = None # URL to a PDF, Word doc, or other supported type
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[ClaimResponseDetail] # Now a list of structured claim response details

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

        # Langchain loaders add metadata like 'page' and 'source'
        if file_extension == "pdf":
            loader = PyPDFLoader(str(temp_file_path))
        elif file_extension in ["doc", "docx"]:
            # Unstructured loaders can be heavy; ensure it's installed if needed
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

# --- Helper function for robust JSON parsing from LLM output ---
def _parse_llm_json_output(text: str) -> Dict[str, Any]:
    """
    Attempts to robustly parse JSON output from an LLM,
    handling common issues like leading/trailing text or markdown blocks.
    """
    # Try to find a JSON block in the text (e.g., ```json...``` or just { ... })
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to decode JSON from LLM output: {e}")
        print(f"LLM output that caused error:\n{text}")
        # Fallback to attempt repair if common issues exist
        try:
            # Try to fix common issues: adding missing brackets, removing trailing commas
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str = json_str + '}'
            # Attempt a more lenient parse (though not always reliable)
            # This is a basic attempt; for production, consider a more robust JSON repair library
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"LLM output is not valid JSON and could not be repaired: {text}")

# --- Helper function to parse claim details ---
async def _parse_claim_details(query: str) -> Dict[str, Any]:
    """
    Uses an LLM to parse natural language claim query into structured details.
    """
    if llm_parser_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM Parser model not initialized.")
    
    # Create the prompt for the parser LLM
    parser_prompt_value = CUSTOM_PARSER_PROMPT.format_prompt(query=query)
    
    try:
        # Invoke the parser LLM
        response = llm_parser_model.invoke(parser_prompt_value.to_string())
        # Parse the JSON output
        parsed_details = _parse_llm_json_output(response.content)
        print(f"Parsed Claim Details: {parsed_details}")
        return parsed_details
    except Exception as e:
        print(f"ERROR: Failed to parse claim details with LLM: {e}")
        return {"processing_notes": [f"Failed to parse claim details: {e}"]}


# --- Application Startup Event (for default policy.pdf) ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components for the default 'policy.pdf'
    and the LLM Parser model once when the FastAPI application starts.
    """
    global default_qa_chain, default_vector_store, llm_parser_model

    print("--- Application Startup: Initializing RAG System ---")

    # Initialize the LLM for parsing claim details (always needed)
    try:
        llm_parser_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=GOOGLE_API_KEY)
        print("LLM Parser model initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM Parser model: {e}")
        # Re-raise to prevent app from starting if critical component fails
        raise

    # Initialize RAG for default policy.pdf (optional, allows dynamic loading if missing)
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

        # Main LLM for claim analysis
        main_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=GOOGLE_API_KEY)
        
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=main_llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(search_kwargs={"k": 8}), # Increased k for more context
            return_source_documents=True, # Essential to get metadata for policy_sections_referenced
            chain_type_kwargs={"prompt": CUSTOM_CLAIM_PROMPT}
        )
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your GOOGLE_API_KEY is correct, 'policy.pdf' exists, and all required packages (like faiss-cpu) are installed.")
        # Do not re-raise here if we want to allow dynamic document handling to proceed


# --- API Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)],
    summary="Run LLM-Powered Claim Processing on Policy Documents"
)
async def run_submission(request_body: QueryRequest):
    """
    Processes a list of natural language questions as insurance claims against the provided document(s) (URL or default)
    and returns structured claim decisions.
    """
    current_vector_store = None
    current_qa_chain = None
    temp_doc_path = None

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if llm_parser_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM Parser is not initialized.")
    if FAISS is None:
        # If FAISS never imported, cannot proceed with any vector store ops
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")

    answers: List[ClaimResponseDetail] = []

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

            main_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=GOOGLE_API_KEY)
            
            current_qa_chain = RetrievalQA.from_chain_type(
                llm=main_llm,
                chain_type="stuff",
                retriever=current_vector_store.as_retriever(search_kwargs={"k": 8}), # Increased k
                return_source_documents=True, # Crucial for metadata
                chain_type_kwargs={"prompt": CUSTOM_CLAIM_PROMPT}
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

        for question in request_body.questions:
            print(f"\nProcessing Claim Question: '{question}'")
            
            # Step 1: Parse the natural language query into structured claim details
            claim_details = await _parse_claim_details(question)
            claim_details_json_str = json.dumps(claim_details, indent=2)

            # Step 2: Invoke the main RAG chain with the structured claim details and original question
            # The 'query' for RetrievalQA will be the original question,
            # and the `CUSTOM_CLAIM_PROMPT` will inject the structured details and context.
            try:
                # result = current_qa_chain.invoke({"query": question})
                # The .run() method takes string arguments directly for simple chains.
                # For complex prompts with multiple input variables, we need to use _call or invoke
                # For RetrievalQA, 'query' is usually the question.
                # We format the prompt inputs manually to ensure `claim_details_json` is passed.
                
                # RetrievalQA.invoke returns {"query": ..., "result": ..., "source_documents": [...]}
                # We need the source_documents for referencing, so we use invoke.
                
                # Fetch relevant docs for the context
                retrieved_docs = current_qa_chain.retriever.get_relevant_documents(question)
                context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Prepare inputs for the custom prompt
                formatted_prompt = CUSTOM_CLAIM_PROMPT.format(
                    context=context_str,
                    claim_details_json=claim_details_json_str,
                    question=question
                )
                
                # Directly invoke the LLM with the fully formatted prompt
                # Note: RetrievalQA.from_chain_type with chain_type="stuff" typically handles this,
                # but if we need full control over prompt structure including retrieved docs
                # AND external variables like claim_details_json, direct LLM call is cleaner.
                # Let's stick to the structure of RetrievalQA but ensure custom prompt handles it.
                # If RetrievalQA setup isn't robust enough for external variables + context,
                # we'd switch to a custom chain or direct LLM.
                
                # Re-configuring RetrievalQA to use multiple inputs requires a different chain type or custom chain.
                # For simplicity and to use the `retriever` component properly, we will modify the prompt slightly.
                # The 'context' is automatically provided by RetrievalQA.
                # We'll pass `claim_details_json` as part of the `query` or rely on the LLM to understand.
                # A better way for multiple inputs is using `RunnableParallel` with `LCEL` or custom chain.
                # For hackathon quick solution, we pass claim_details_json as part of question.
                
                # Let's simplify the RetrievalQA usage for now and assume the LLM can handle the JSON in the prompt
                # without requiring it as a separate `input_variable` to RetrievalQA itself.
                # The context comes from retriever, question is the query. We'll inject claim_details_json into prompt template.
                
                # The correct way to pass `claim_details_json` is through the chain_type_kwargs
                # which is already set for CUSTOM_CLAIM_PROMPT. So, the original `qa_chain.invoke` should work.
                
                result = current_qa_chain.invoke({
                    "query": question, # This goes into {question} in prompt
                    "claim_details_json": claim_details_json_str # This goes into {claim_details_json} in prompt
                })

                llm_raw_output = result.get("result", "")
                
                # Extract policy sections and exclusions from source documents
                policy_sections = []
                exclusions_applied = []
                source_documents = result.get("source_documents", [])
                
                for doc in source_documents:
                    source_str = f"Document: {Path(doc.metadata.get('source', 'unknown_doc')).name}"
                    if 'page' in doc.metadata:
                        source_str += f", Page {doc.metadata['page'] + 1}" # Page numbers are often 0-indexed
                    
                    # This is a basic attempt. LLM might reference clause numbers from text directly.
                    # We assume LLM explicitly calls out "Section X", "Clause Y" etc. in its reasoning
                    # and we just append the document source to every retrieved document.
                    # More sophisticated parsing of LLM's 'reasoning' to map specific clauses needed for full automation.
                    policy_sections.append(source_str)
                    # For exclusions, a more advanced regex on reasoning might be needed to identify if an exclusion was applied.
                    # For now, we'll assume the LLM will explicitly name exclusions in reasoning.

                # Parse the LLM's structured JSON output
                parsed_response_dict = _parse_llm_json_output(llm_raw_output)

                # Fill in source documents if LLM didn't (or refine if LLM did)
                if not parsed_response_dict.get("policy_sections_referenced"):
                    parsed_response_dict["policy_sections_referenced"] = list(set(policy_sections)) # Deduplicate
                
                # Attempt to parse into Pydantic model
                claim_response = ClaimResponseDetail(**parsed_response_dict)
                
            except Exception as e:
                print(f"ERROR: Failed to process claim with RAG chain or parse LLM output: {e}")
                # Create a fallback error response
                claim_response = ClaimResponseDetail(
                    decision="PENDING_REVIEW",
                    confidence_score=0.0,
                    payout_amount=None,
                    reasoning=f"Failed to process claim due to internal error: {e}",
                    policy_sections_referenced=[],
                    exclusions_applied=[],
                    coordination_of_benefits=CoordinationOfBenefits(has_other_insurance=False),
                    processing_notes=[f"Error during main claim processing: {e}"]
                )
            answers.append(claim_response)
            print(f"Claim response generated for '{question}'.")

        print("--- All questions processed. Sending structured response. ---")
        return {"answers": answers}

    except HTTPException:
        # Re-raise HTTPExceptions directly as they are already formatted
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during overall query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        # Clean up temporary document if one was downloaded
        if temp_doc_path and temp_doc_path.exists():
            try:
                os.remove(temp_doc_path)
                print(f"Cleaned up temporary file: {temp_doc_path}")
                if not any(Path("./temp_docs").iterdir()): # Remove dir if empty
                    os.rmdir("./temp_docs")
            except Exception as cleanup_e:
                print(f"WARNING: Failed to clean up temporary file {temp_doc_path}: {cleanup_e}")

# --- Root Endpoint (Optional, for quick health check) ---
@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "LLM-Powered Intelligent Claim Processing System API is running. Visit /api/v1/docs for interactive documentation."}
