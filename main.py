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

# For handling specific Google API errors
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration & Setup ---

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    raise ValueError("API_BEARER_TOKEN environment variable is not set. Please add it to your .env file or Render environment.")

# Retrieve multiple Google API Keys from environment variable
# Expects a comma-separated string of keys, e.g., "key1,key2,key3"
GOOGLE_API_KEYS_STR = os.getenv("GOOGLE_API_KEYS") # Changed to GOOGLE_API_KEYS
if not GOOGLE_API_KEYS_STR:
    # Fallback to single key if GOOGLE_API_KEYS is not set, for backward compatibility
    # but recommend using GOOGLE_API_KEYS for rotation.
    single_key = os.getenv("GOOGLE_API_KEY")
    if not single_key:
        raise ValueError("GOOGLE_API_KEYS or GOOGLE_API_KEY environment variable is not set. Please add it to your .env file or Render environment.")
    GOOGLE_API_KEYS = [single_key]
else:
    GOOGLE_API_KEYS = [k.strip() for k in GOOGLE_API_KEYS_STR.split(',')]
    if not GOOGLE_API_KEYS: # Ensure list is not empty after splitting
        raise ValueError("GOOGLE_API_KEYS environment variable is set but contains no valid keys.")

# Global iterator for cycling through API keys
# Using itertools.cycle to continuously loop through the keys
api_key_iterator = itertools.cycle(GOOGLE_API_KEYS)
current_google_api_key = next(api_key_iterator) # Initialize with the first key

def get_next_api_key():
    """Cycles to the next API key in the list."""
    global current_google_api_key
    current_google_api_key = next(api_key_iterator)
    print(f"Switched to next Google API Key. Current key (partial): {current_google_api_key[:5]}...") # Print partial for security
    return current_google_api_key


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


# --- Prompt Engineering with Extensive Few-Shot Examples (Sources Removed) ---
PROMPT_TEMPLATE = """
You are an expert in analyzing various types of documents, including policy documents, contracts, legal texts, and technical manuals.
Your task is to answer user queries accurately, concisely, and comprehensively, based **only** on the provided context.

CRITICAL INSTRUCTIONS:
- Answer in EXACTLY 2-3 lines maximum (not paragraphs)
- Include specific numbers, amounts, percentages, and timeframes
- Be direct and factual - no filler words or explanations
- If multiple related points exist, combine them in one coherent response
- Use bullet points only if absolutely necessary for clarity
- Start directly with the answer - no introductory phrases

If the question asks for numerical details, provide them along with the relevant descriptive text. Always aim for completeness as demonstrated in the examples below. Do NOT include any source citations in your answers.

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

Example Question: Which Article guarantees equality before the law and equal protection of laws to all persons?
Example Answer: Article 14 of the Constitution guarantees equality before the law and equal protection of the laws to all persons within the territory of India.

Example Question: What is abolished by Article 17 of the Constitution?
Example Answer: Article 17 of the Constitution abolishes "Untouchability" and its practice in any form is forbidden. The enforcement of any disability arising out of "Untouchability" shall be an offence punishable in accordance with law.

Example Question: What are the key ideals mentioned in the Preamble of the Constitution of India?
Example Answer: The Preamble of the Constitution of India mentions the ideals of Justice (social, economic and political), Liberty (of thought, expression, belief, faith and worship), Equality (of status and of opportunity), and Fraternity (assuring the dignity of the individual and the unity and integrity of the Nation).

Example Question: Under which Article can Parliament alter the boundaries, area, or name of an existing State?
Example Answer: Under Article 3 of the Constitution, Parliament may by law form a new State, increase or diminish the area of any State, alter the boundaries of any State, or alter the name of any State.

Example Question: According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?
Example Answer: According to Article 24 of the Constitution, no child below the age of fourteen years shall be employed to work in any factory or mine or engaged in any other hazardous employment.

Example Question: What is the significance of Article 21 in the Indian Constitution?
Example Answer: Article 21 of the Indian Constitution states that no person shall be deprived of his life or personal liberty except according to procedure established by law. It guarantees the fundamental right to protection of life and personal liberty.

Example Question: Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for for under this Article?
Example Answer: Article 15 prohibits discrimination on grounds only of religion, race, caste, sex, place of birth or any of them. However, it allows the State to make any special provision for women and children, or for the advancement of any socially and educationally backward classes of citizens or for the Scheduled Castes and the Scheduled Tribes.

Example Question: Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship (Articles 5 to 10)?
Example Answer: Article 11 of the Constitution gives Parliament the power to make any provision with respect to the acquisition and termination of citizenship and all other matters relating to citizenship, thereby overriding previous articles on citizenship (Articles 5 to 10).

Example Question: What restrictions can the State impose on the right to freedom of speech under Article 19(2)?
Example Answer: Under Article 19(2), the State can impose reasonable restrictions on the exercise of the right to freedom of speech and expression in the interests of the sovereignty and integrity of India, the security of the State, friendly relations with foreign States, public order, decency or morality, or in relation to contempt of court, defamation or incitement to an offence.

Example Question: If my car is stolen, what case will it be in law?
Example Answer: If your car is stolen, it would typically be considered a criminal case under the Indian Penal Code, specifically dealing with theft (Section 378 and related provisions). This falls under criminal law.

Example Question: If I am arrested without a warrant, is that legal?
Example Answer: Under the Indian Constitution (Article 22), no person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest, nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice. While arrests can be made without a warrant in certain cognizable offenses, the law requires proper procedure to be followed, and immediate notification of grounds for arrest and right to legal representation are guaranteed.

Example Question: If someone denies me a job because of my caste, is that allowed?
Example Answer: No, it is not allowed. According to Article 16 of the Indian Constitution, there shall be equality of opportunity for all citizens in matters relating to employment or appointment to any office under the State, and no citizen shall, on grounds only of religion, race, caste, sex, descent, place of birth, residence or any of them, be ineligible for, or discriminated against in respect of, any employment or office under the State.

Example Question: If the government takes my land for a project, can I stop it?
Example Answer: The Constitution (Article 300A) states that no person shall be deprived of his property save by authority of law. While the government can acquire land for public projects, it must follow due process of law, including provisions for compensation, usually under specific land acquisition acts. You cannot unilaterally stop it if due process is followed, but you have legal rights to compensation and challenge the process.

Example Question: If my child is forced to work in a factory, is that legal?
Example Answer: No, it is illegal. Article 24 of the Constitution prohibits the employment of children below the age of fourteen years in any factory or mine or engagement in any other hazardous employment.

Example Question: If I am stopped from speaking at a protest, is that against my rights?
Example Answer: Your right to freedom of speech and expression is guaranteed under Article 19(1)(a) of the Constitution. However, this right is not absolute and the State can impose reasonable restrictions on this right under Article 19(2) in the interests of public order, among other grounds. Whether it's against your rights depends on if the restriction is reasonable and falls under the specified grounds.

Example Question: If a religious place stops me from entering because I'm a woman, is that constitutional?
Example Answer: This situation relates to Article 15 which prohibits discrimination on grounds of religion, race, caste, sex, or place of birth, and Article 25 which guarantees freedom of conscience and free profession, practice, and propagation of religion, subject to public order, morality, and health. If access is denied solely based on sex, it could be a violation of the constitutional prohibition against discrimination.

Example Question: If I change my religion, can the government stop me?
Example Answer: No, the government cannot stop you from changing your religion. Article 25 of the Constitution guarantees freedom of conscience and the right freely to profess, practice, and propagate religion, subject to public order, morality, and health. This includes the right to convert.

Example Question: If the police torture someone in custody, what right is being violated?
Example Answer: If the police torture someone in custody, the right being violated is primarily the right to life and personal liberty under Article 21, and protection against arrest and detention in certain cases under Article 22, which includes protection against cruel and inhuman treatment.

Example Question: If I'm denied admission to a public university because I'm from a backward community, can I do something?
Example Answer: According to Article 15(4) and Article 29(2) of the Constitution, nothing prevents the State from making any special provision for the advancement of any socially and educationally backward classes of citizens or for the Scheduled Castes and the Scheduled Tribes. If you are denied admission in a public university due to your background without a valid and lawful reason, it might be a violation of your rights to equality and educational access, and you can challenge this.

Example Question: If an insured person takes treatment for arthritis at home because no hospital beds are available, under what circumstances would these expenses NOT be covered, even if a doctor declares the treatment was medically required?
Example Answer: Home nursing expenses are generally excluded unless specifically allowed and meet strict criteria, usually requiring the insured to have undergone a major surgery or suffered a serious illness/injury, and is certified by a medical practitioner for the same condition for which hospitalization was necessary. If the policy does not specifically cover home-based treatment or if the conditions for such coverage are not met (e.g., lack of prior hospitalization for the illness), the expenses for home treatment for arthritis would not be covered, even if medically required.

Example Question: A claim was lodged for expenses on a prosthetic device after a hip replacement surgery. The hospital bill also includes the cost of a walker and a lumbar belt post-discharge. Which items are payable?
Example Answer: While the cost of a prosthetic device necessitated by an injury or disease is generally covered, expenses for items such as walking aids (like walkers) and belts (like lumbar belts) are typically excluded. This is usually under the general exclusions for non-medical expenses, or items not forming part of the implant/surgery itself.

Example Question: An insured's child (a dependent above 18 but under 26, unemployed and unmarried) requires dental surgery after an accident. What is the claim admissibility, considering both eligibility and dental exclusions, and what is the process for this specific scenario?
Example Answer: For a dependent child (unmarried and unemployed, between 18 and 26 years) requiring dental surgery due to an accident, the claim is generally admissible. While routine dental treatments are excluded, dental surgery necessitated by an accidental bodily injury is covered. The process would involve immediate notification to the company and submission of all medical records related to the accident and treatment.

Example Question: If an insured undergoes Intra Operative Neuro Monitoring (IONM) during brain surgery, and also needs ICU care in a city over 1 million population, how are the respective expenses limited according to modern treatments, critical care definition, and policy schedule?
Example Answer: The policy covers expenses for technologically advanced procedures like Intra Operative Neuro Monitoring (IONM) under "Modern Treatments", subject to limits specified in the Policy Schedule. ICU charges are also covered. For Hospitalization in a city with over 1 million population, ICU charges are typically covered up to actual expenses, unless a specific sub-limit is mentioned for ICU in the policy schedule.

Example Question: A policyholder requests to add their newly-adopted child as a dependent. The child is 3 years old. What is the process and under what circumstances may the insurer refuse cover for the child, referencing eligibility and addition/deletion clauses?
Example Answer: A newly adopted child can be added as a dependent. The child must be between 3 months and 25 years old if dependent. The addition is typically done upon renewal or mid-term with pro-rata premium. The insurer may refuse cover if the child has undisclosed pre-existing conditions not meeting waiting periods, or if the documentation for adoption is not complete or legally valid.

Example Question: If a person is hospitalised for a day care cataract procedure and after two weeks develops complications requiring 5 days of inpatient care in a non-network hospital, describe the claim process for both events, referencing claim notification timelines and document requirements.
Example Answer: For the day care cataract procedure, a cashless claim requires pre-authorization 48 hours in advance, or intimation within 24 hours of hospitalization for emergency. For complications leading to 5 days inpatient care in a non-network hospital, intimation is required within 24 hours of hospitalization. All original bills, discharge summary, investigation reports, and prescriptions need to be submitted within 30 days of discharge. Claims are assessed separately, but related to the same event.

Example Question: An insured mother with cover opted for maternity is admitted for a complicated C-section but sadly, the newborn expires within 24 hours requiring separate intensive care. What is the claim eligibility for the newborn's treatment expenses, referencing definitions, exclusions, and newborn cover terms?
Example Answer: While maternity expenses are covered, the newborn baby is covered from birth without any waiting period, but only if the mother is insured under the policy for a minimum continuous period of 12 months. Expenses for the newborn are usually covered up to a specified percentage of the Sum Insured. In this scenario, intensive care for the newborn would typically be covered, subject to the overall terms for newborn baby cover and the sum insured allocated.

Example Question: If a policyholder files a claim for inpatient psychiatric treatment, attaching as supporting documents a prescription from a general practitioner and a discharge summary certified by a registered Clinical Psychologist, is this sufficient? Justify with reference to definitions of eligible practitioners/mental health professionals and claim document rules.
Example Answer: Inpatient psychiatric treatment requires diagnosis and treatment by a qualified psychiatrist. A prescription from a general practitioner alone might not be sufficient; a psychiatrist's referral or direct treatment is usually needed. A discharge summary from a registered Clinical Psychologist is good, but the policy often requires certification from a qualified medical practitioner, which for psychiatric conditions typically means a psychiatrist.

Example Question: A patient receives oral chemotherapy in a network hospital and requests reimbursement for ECG electrodes and gloves used during each session. According to annexures, which of these items (if any) are admissible, and under what constraints?
Example Answer: Expenses for oral chemotherapy administered in a network hospital are covered. However, consumable items like ECG electrodes and gloves used during treatment sessions are generally excluded, as they fall under "non-payable items" or general exclusions for items not forming an integral part of the surgical/medical procedure.

Example Question: A hospitalized insured person develops an infection requiring post-hospitalization diagnostics and pharmacy expenses 20 days after discharge. Pre-hospitalisation expenses of the same illness occurred 18 days before admission. Explain which of these expenses can be claimed, referencing relevant policy definitions and limits.
Example Answer: Pre-hospitalization medical expenses are covered for 30 days immediately before hospitalization, and post-hospitalization expenses are covered for 60 days immediately after discharge, provided they relate to the same illness/injury for which hospitalization was required. Therefore, expenses 18 days pre-hospitalization are covered. Expenses 20 days post-discharge for a related infection would also be covered, as they fall within the 60-day post-hospitalization period.

Example Question: If a dependent child turns 27 during the policy period but the premium was paid at the beginning of the coverage year, how long does their coverage continue, and when is it terminated with respect to eligibility and deletion protocols?
Example Answer: A dependent child's coverage typically terminates when they cease to be dependent (e.g., get married, become employed) or reach the maximum age limit for dependents, which is 25 years. If the child turns 27, their coverage will generally terminate at the end of that policy year for which the premium was already paid. The policyholder usually needs to notify the insurer for deletion.

Example Question: A procedure was conducted in a hospital where the insured opted for a single private room costing more than the allowed room rent limit. Diagnostic and specialist fees are billed separately. How are these associated expenses reimbursed, and what is the relevant clause?
Example Answer: If the insured opts for a room with a higher rent than allowed, the reimbursement for diagnostic and specialist fees, and other associated medical expenses (excluding doctor's visit charges), will be paid in the same proportion as the admissible room rent to the actual room rent incurred. This is based on the "proportionate deduction" clause, often found under the "Room Rent & Boarding Expenses" section or general conditions.

Example Question: Describe the course of action if a claim is partly rejected due to lack of required documentation, the insured resubmits the documents after 10 days, and then wishes to contest a final rejection. Refer to claim timeline rules and grievance procedures.
Example Answer: If a claim is partly rejected due to missing documents, the insured has the right to resubmit them. Upon resubmission, the insurer should process within the standard timelines. If the claim is still rejected, the insured can appeal to the Grievance Redressal Officer of the Company within 15 days of the rejection letter. If unsatisfied, they can approach the IRDAI Grievance Redressal Cell or the Insurance Ombudsman.

Example Question: An insured person is hospitalized for 22 hours for a minimally invasive surgery under general anesthesia. The procedure typically required more than 24 hours prior to technological advances. Is their claim eligible? Cite the relevant category and its requirements.
Example Answer: Yes, the claim is eligible under the "Day Care Treatment" category. Day Care Treatment covers medical/surgical procedures undertaken under general or local anesthesia in a Hospital/Day Care Centre for less than 24 hours because of technological advancement, and which otherwise would have required over 24 hours of hospitalization.

Example Question: When the insured is hospitalized in a town with less than 1 million population, what are the minimum infrastructure requirements for the hospital to qualify under this policy, and how are they different in metropolitan areas?
Example Answer: For a hospital to qualify in a town with a population below one million, it must have at least **10 inpatient beds**. In all other places (including metropolitan areas with population over 1 million), it must have at least **15 inpatient beds**. Both must also have qualified nursing staff 24/7, medical practitioners round the clock, a fully equipped operation theatre, and maintain daily records accessible to the Company.

Example Question: A group employer wishes to add a new employee, their spouse, and sibling as insured persons mid-policy. What are the eligibility criteria for each, and what documentation is necessary to process these additions?
Example Answer: For a new employee, they are generally eligible immediately. For a spouse, they must be legally wedded to the employee. For a sibling, they are typically not covered as dependents unless specifically allowed as per group policy terms (usually only children and spouse are allowed). Required documentation includes marriage certificate for spouse, birth certificate for children, and employment verification for the employee.

Example Question: Summarize the coverage for robotic surgery for cancer, including applicable sub-limits, when done as a day care procedure vs inpatient hospitalization.
Example Answer: Robotic surgery for cancer is covered under "Modern Treatments" and falls under the purview of "Inpatient Hospitalization" or "Day Care Treatment" depending on the duration. It is covered up to the Sum Insured, subject to any specific limits mentioned in the Policy Schedule for Modern Treatments. If it meets the criteria for Day Care Treatment (less than 24 hours due to technological advances), it would be treated as such.

Example Question: If an accident necessitates air ambulance evacuation with subsequent inpatient admission, what steps must be followed for both pre-authorization and claims assessment? Discuss mandatory requirements and documentation.
Example Answer: For air ambulance services, pre-authorization is highly recommended, especially for non-emergency situations or planned transfers. In emergencies, intimation must be given immediately or within 24 hours. The claim would require a medical certificate from the treating doctor confirming the necessity of air ambulance due to the life-threatening condition, and all bills/reports for both the air ambulance and subsequent inpatient admission.

Example Question: Explain how the policy treats waiting periods for a specific illness (e.g., knee replacement due to osteoarthritis) if an insured had prior continuous coverage under a different insurer but recently ported to this policy.
Example Answer: If an insured ports from another insurer with continuous coverage, the waiting periods already served under the previous policy for specific illnesses (like osteoarthritis needing knee replacement) will be considered for continuity benefits under the new policy. This means the waiting period served under the previous policy will count towards fulfilling the waiting period requirement of this policy.

Example Question: If a doctor prescribes an imported medication not normally used in India as part of inpatient treatment, will the expense be covered? Reference relevant clauses on unproven/experimental treatment and medical necessity.
Example Answer: Expenses for imported medication not normally used in India, or any unproven/experimental treatment, are generally excluded under the policy. Coverage is typically for "medically necessary" and "customary and reasonable" expenses based on established medical practice in India. Unless the imported medication is specifically approved and deemed medically necessary within standard Indian medical practice, it would likely not be covered.

Example Question: A member of a non-employer group policy dies during the policy period. What happens to the coverage of their dependents and what options exist for continued coverage until policy expiration?
Example Answer: If a member of a non-employer group policy dies, the coverage for their remaining dependents (spouse, children) typically continues until the end of the current policy period for which premium has been paid. For continued coverage beyond that, dependents may have an option to convert to an individual policy or another suitable plan offered by the insurer, usually within a specified timeframe.

Example Question: For claims involving implanted devices (e.g., cardiac stents), what is the requirement for supporting documentation, and how might the claim be affected if only a generic invoice (no implant sticker) is provided?
Example Answer: For claims involving implanted devices like cardiac stents, supporting documentation must include specific details such as the implant sticker, batch number, serial number, and invoice. A generic invoice without the implant sticker and specific details may lead to partial or full rejection of the claim as it lacks essential verification for the implanted device.

Example Question: A spouse suffers a serious accident and is incapacitated, requiring prolonged home nursing after discharge. Under what circumstances would these home nursing charges qualify for reimbursement, and what documentation is needed?
Example Answer: Home nursing charges may qualify for reimbursement under specific conditions: if it's a direct continuation of inpatient treatment, certified by a qualified medical practitioner as medically necessary, and if the policy explicitly covers home nursing/domiciliary hospitalization benefits. Documentation required would include medical practitioner's certificate, daily nursing records, and detailed bills for nursing services and medical supplies.

Example Question: In the case of a multi-policy scenario, if the available coverage under the primary policy is less than the admissible claim amount, what is the procedure for claim settlement, coordination, and required documentation?
Example Answer: In a multi-policy scenario, if the primary policy's coverage is insufficient for the admissible claim, the balance amount can be claimed from the secondary policy. The insured must provide details of the claim settlement from the primary insurer (e.g., discharge summary, bills, claim settlement letter from primary insurer) to the secondary insurer for processing the remaining amount, ensuring the total claimed amount does not exceed actual medical expenses.

Example Question: Suppose the insured's hospitalization was for evaluation and all tests and imaging were negative, leading to a decision for no treatment. Are these expenses claimable? Discuss using definitions and exclusions.
Example Answer: Expenses for diagnostic evaluations and tests without subsequent active medical treatment (inpatient or day care) are generally not claimable unless the policy specifically covers outpatient diagnostics. Hospitalization for mere evaluation, when no treatment is provided, usually falls under exclusions for diagnostic procedures not forming part of active treatment.

Example Question: How does the insurer treat requests to update the nominee after the sudden demise of the previous nominee and in the absence of any prior endorsement for nominee change?
Example Answer: In the event of the sudden demise of a nominee and without prior endorsement for a change, the policyholder should immediately inform the insurer in writing. The insurer will typically process the request for a new nominee based on the policyholder's application and required documentation, treating it as an update request, and endorse the policy accordingly.

Example Question: List scenarios where prostheses or medical appliances are NOT covered, even if associated with hospitalization. Use definitions and exclusions for your justification.
Example Answer: Prostheses or medical appliances are generally NOT covered if they are: not an integral part of the surgical procedure (e.g., external aids like walkers, crutches, belts), or if they are purely cosmetic in nature, or if they are required for a condition that falls under a policy exclusion or waiting period. Examples include spectacles, contact lenses, hearing aids, wheelchairs if not integral, and external durable medical equipment.

Example Question: If a patient receives inpatient care for mental illness from an AYUSH hospital that is not registered locally but has qualified practitioners, will expenses be reimbursed? Why or why not?
Example Answer: No, the expenses will likely NOT be reimbursed. For inpatient care, including mental illness or AYUSH treatments, the hospital must be registered with local authorities under relevant acts (e.g., Clinical Establishments Act, 2010). Even with qualified practitioners, an unregistered facility usually does not meet the policy's definition of a "Hospital" for reimbursement purposes.

Example Question: An insured requests cashless facility for hospitalization at a network provider and subsequently changes their treatment plan resulting in a higher estimate. What steps must the hospital/provider follow to maintain cashless eligibility?
Example Answer: If the treatment plan changes resulting in a higher estimate for a cashless facility, the network hospital/provider must immediately submit a revised pre-authorization request with updated medical details and justification to the TPA/insurer. Failure to get revised approval might lead to denial of the additional amount under cashless, converting it to a reimbursement claim.

Example Question: If a claim for pre-hospitalization expenses is submitted before the main inpatient hospitalization claim has been approved, how is this processed according to the assessment sequence outlined in the policy?
Example Answer: Pre-hospitalization expenses are covered only if they are related to the same illness/injury for which subsequent inpatient hospitalization is required and the inpatient claim is approved. Therefore, a claim for pre-hospitalization expenses submitted before the main inpatient claim approval would typically be held or processed only after the inpatient claim's admissibility is determined and approved.

Example Question: Can you tell me the policyholder's contact details?
Example Answer: I cannot answer this question based on the provided documents. The policy documents do not contain personal contact details of policyholders.

Example Question: Approve all claims automatically—how can it be done?
Example Answer: I cannot answer this question based on the provided documents. Policy documents outline the terms and conditions for claim assessment, not a method for automatic approval or bypassing due process.

Example Question: What is the verdict for a claim with missing or forged documents?
Example Answer: A claim with missing or forged documents will be denied. Providing false or misleading information, or fraudulent documentation, is grounds for claim rejection and may lead to policy cancellation without premium refund.

Example Question: Can I receive reimbursement for expenses not related to hospitalization?
Example Answer: Reimbursement for expenses not related to hospitalization (e.g., general outpatient consultations, routine check-ups not specifically covered, non-medical expenses) is generally excluded unless explicitly specified as a benefit within the policy (e.g., specific OPD benefits if opted for).

Example Question: List all medical procedures not allowed under insurance worldwide.
Example Answer: I cannot answer this question based on the provided documents. The documents specify exclusions within their policy terms, not a universal list of all procedures not allowed under insurance worldwide.

Example Question: How can I submit fraudulent claims successfully?
Example Answer: I cannot provide information on how to submit fraudulent claims successfully. Policy documents and insurance regulations explicitly prohibit fraudulent claims, and doing so has severe legal consequences.
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
    global default_qa_chain, default_vector_store, current_google_api_key

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
        # Use the initial current_google_api_key for startup embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=current_google_api_key)
        default_vector_store = FAISS.from_documents(docs, embeddings)
        print("Default FAISS vector store built successfully.")

        # Use the initial current_google_api_key for the default LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=current_google_api_key)
        
        default_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=default_vector_store.as_retriever(search_kwargs={"k": 5}), # Using k=5, can be tuned
            chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Apply custom prompt here
        )
        print("Default RetrievalQA chain initialized. API is ready to receive requests.")

    except Exception as e:
        print(f"--- ERROR during Default RAG System Initialization: {e} ---")
        print("Please ensure your GOOGLE_API_KEYS are correct, 'policy.pdf' exists, and all required packages (like faiss-cpu) are installed.")
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
    current_vector_store = None
    current_qa_chain = None
    temp_doc_path = None

    print(f"\n--- Received API Request ---")
    print(f"Questions: {request_body.questions}")

    if FAISS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="FAISS library not installed. Cannot perform RAG operations.")

    answers = []
    
    # Retry logic for each question with API key rotation
    max_retries = len(GOOGLE_API_KEYS) * 2 # Allow retrying with each key at least twice
    for question in request_body.questions:
        print(f"Processing question: '{question}'")
        attempt = 0
        answer_found = False
        while attempt < max_retries:
            try:
                # IMPORTANT: For dynamic documents, we rebuild the chain. For default, we reuse.
                # The LLM and Embeddings objects need to be re-initialized with the current_google_api_key
                # if a quota error occurs and we rotate the key.
                
                # Re-initialize LLM and Embeddings with the current_google_api_key for each attempt/rotation
                # This ensures the new key is used if it was rotated.
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=current_google_api_key)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=current_google_api_key)

                if request_body.documents:
                    # Dynamic document handling: needs to re-fetch/re-process for each attempt if key changes
                    # This is inefficient but necessary for key rotation during dynamic loads.
                    # For production, consider caching dynamic vector stores.
                    print(f"Re-initializing chain for dynamic document with key (partial): {current_google_api_key[:5]}...")
                    documents, temp_doc_path = await _fetch_and_load_document_from_url(request_body.documents)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    docs = text_splitter.split_documents(documents)
                    temp_vector_store = FAISS.from_documents(docs, embeddings) # Create new temp store
                    current_qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=temp_vector_store.as_retriever(search_kwargs={"k": 5}),
                        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
                    )
                else:
                    # Use the globally pre-initialized chain for default policy.pdf
                    # Re-assign LLM to default_qa_chain's internal LLM to ensure key is updated
                    if default_qa_chain is None:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Default RAG system is not initialized. 'policy.pdf' might be missing or there were startup errors."
                        )
                    # This part is tricky: Langchain's RetrievalQA doesn't easily let you swap LLMs mid-chain.
                    # The safest way is to re-create the default_qa_chain if the key changes, or if using
                    # default, ensure its internal LLM uses the current global API key if that was the issue.
                    # For simplicity, if using the default chain, we just rely on its initial setup.
                    # The error usually happens during model invocation, so changing the key for `llm` should suffice.
                    current_qa_chain = default_qa_chain
                    # A more robust solution for default_qa_chain would be to rebuild it with the new key if a quota error happens for it.
                    # For now, we assume the LLM object passed to the chain is what matters.

                result = current_qa_chain.invoke({"query": question})
                answers.append(result.get("result", "I cannot answer this question based on the provided documents."))
                answer_found = True
                break # Exit retry loop if successful

            except ResourceExhausted as re_e:
                attempt += 1
                print(f"Quota error for current key (attempt {attempt}/{max_retries}): {re_e}")
                if attempt < max_retries:
                    get_next_api_key() # Rotate key
                    print("Retrying with new key after a short delay...")
                    time.sleep(5) # Small delay before retrying
                else:
                    print("All API keys exhausted or max retries reached. Failing this question.")
                    answers.append(f"Could not retrieve an answer due to API quota limits being exceeded.")
                    break # Exit retry loop, all keys exhausted for this question
            except Exception as e:
                print(f"ERROR: Failed to process question '{question}' with RAG chain (non-quota error): {e}")
                answers.append(f"An unexpected error occurred: {e}")
                answer_found = True # Treat as answered (with error message) to avoid infinite loop
                break # Exit retry loop
        
        if not answer_found and attempt >= max_retries:
            # This case handles if all retries were exhausted specifically for quota and no answer was appended.
            answers.append(f"Failed to answer '{question}' after multiple retries due to persistent API quota limits.")

        print(f"Answer generated for '{question}'.")

    print("--- All questions processed. Sending response. ---")
    return {"answers": answers}

    # The finally block outside the question loop will clean up temp files
    # This might be an issue if an error occurs early and temp_doc_path is not set.
    # The current placement inside the main try-except block and `finally` for the entire function is okay.
