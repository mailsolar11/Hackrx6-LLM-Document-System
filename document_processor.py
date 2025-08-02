import os
from PyPDF2 import PdfReader
from docx import Document
from email.parser import BytesParser, Parser
from email.policy import default

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import VECTOR_DB_PATH, GOOGLE_API_KEY

# Initialize embedding model for Gemini
# 'models/embedding-001' is a stable, supported embedding model ID.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Initialize LLM for Gemini with a supported model
# Use a specific, versioned model like 'gemini-1.5-flash'
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# Define Pydantic output model for structured response
class DecisionOutput(BaseModel):
    Decision: str = Field(description="The final decision (e.g., 'Approved', 'Rejected', 'Needs More Info').")
    Amount: float | None = Field(description="The payout amount if applicable, otherwise null.", default=None)
    Justification: str = Field(description="Detailed explanation for the decision, referencing specific clauses.")
    RelevantClauses: list[dict[str, str]] = Field(description="List of clauses or rules that informed the decision, with their source filename and content snippet.")

# --- Document parsing functions (no changes needed here) ---
def parse_document(filepath, file_extension):
    content = ""
    if file_extension == 'pdf':
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
    elif file_extension == 'docx':
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
    elif file_extension == 'txt':
        loader = TextLoader(filepath)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
    elif file_extension == 'eml':
        with open(filepath, 'rb') as fp:
            msg = BytesParser(policy=default).parse(fp)
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = part.get('Content-Disposition')
                    if ctype == 'text/plain' and (cdisp is None or cdisp == 'inline'):
                        content = part.get_payload(decode=True).decode()
                        break
            else:
                content = msg.get_payload(decode=True).decode()
    else:
        raise ValueError("Unsupported file type")
    return content

def chunk_document(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents([text])
    return chunks

def get_vector_store(chunks):
    """
    Creates or updates a FAISS vector store.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
    else:
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.save_local(VECTOR_DB_PATH)
    return vector_store