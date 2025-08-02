# --- Requirements ---
# Before running this code, install the necessary libraries with pip:
# pip install Flask python-dotenv langchain langchain-community langchain-google-genai pydantic PyPDF2 python-docx requests pandas lxml faiss-cpu Flask-Cors

import os
import json
import requests
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from email.parser import BytesParser, Parser
from email.policy import default

# --- LangChain and Pydantic Imports ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- New Imports for Multi-Step RAG ---
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

# --- Configuration (can be moved to config.py if desired) ---
# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'eml', 'csv', 'json', 'html'}
# Gemini API Key loaded from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# In-memory storage for processed documents' metadata
DOCUMENTS_DB = {}
VECTOR_DB_PATH = "faiss_index"

# --- LLM and Embedding Model Initialization ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# Define Pydantic output model for structured response
class DecisionOutput(BaseModel):
    Decision: str = Field(description="The final decision (e.g., 'Approved', 'Rejected', 'Needs More Info').")
    Amount: float | None = Field(description="The payout amount if applicable, otherwise null.", default=None)
    Justification: str = Field(description="Detailed explanation for the decision, referencing specific clauses.")
    RelevantClauses: list[dict[str, str]] = Field(description="List of clauses or rules that informed the decision, with their source filename and content snippet.")

# --- Document Processing Functions ---
def parse_document(filepath, file_extension):
    """Parses a document of a given type and returns its content as a single string."""
    docs = []
    content = ""
    
    if file_extension == 'pdf':
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif file_extension == 'docx':
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
    elif file_extension == 'txt':
        loader = TextLoader(filepath)
        docs = loader.load()
    elif file_extension == 'csv':
        loader = CSVLoader(filepath)
        docs = loader.load()
    elif file_extension == 'json':
        loader = JSONLoader(filepath, jq_schema=".[]")
        docs = loader.load()
    elif file_extension == 'html':
        loader = UnstructuredHTMLLoader(filepath)
        docs = loader.load()
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
        return content

    if docs:
        content = "\n".join([doc.page_content for doc in docs])
        return content
    else:
        return ""

def chunk_document(text: str):
    """Splits a document text into smaller, overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents([text])
    return chunks

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global vector store
global_vector_store = None
if os.path.exists(VECTOR_DB_PATH):
    print("Loading existing vector store on startup...")
    try:
        global_vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}. Starting with an empty one.")
        global_vector_store = None
else:
    print("No existing FAISS index found. Starting with an empty vector store.")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Routes ---

@app.route('/')
def api_status():
    """A simple status endpoint for the API."""
    return jsonify({"status": "LLM Document Processing API is running."})

@app.route('/process_document', methods=['POST'])
def process_document_api():
    """
    Unified API endpoint to handle document ingestion from a URL or a local file path.
    The request body must be JSON with either a 'url' or 'filepath' key.
    """
    global global_vector_store
    
    filename = None
    filepath = None
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        if 'url' in data:
            url = data['url']
            if not url:
                return jsonify({"error": "URL cannot be empty"}), 400

            url_path = os.path.basename(url.split('?')[0])
            filename = f"{uuid.uuid4()}_{secure_filename(url_path)}"
            if not allowed_file(filename):
                return jsonify({"error": f"File type '{filename.rsplit('.', 1)[1]}' from URL is not allowed"}), 400
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

        elif 'filepath' in data:
            local_path = data['filepath']
            if not local_path or not os.path.exists(local_path):
                return jsonify({"error": "Local filepath does not exist"}), 400

            filename = secure_filename(os.path.basename(local_path))
            if not allowed_file(filename):
                return jsonify({"error": "File type from local path is not allowed"}), 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(local_path, 'rb') as src_file, open(filepath, 'wb') as dest_file:
                dest_file.write(src_file.read())

        else:
            return jsonify({"error": "No 'url' or 'filepath' provided in JSON body"}), 400

        file_extension = filename.rsplit('.', 1)[1].lower()
        doc_content = parse_document(filepath, file_extension)
        chunks = chunk_document(doc_content)
        
        doc_id = len(DOCUMENTS_DB) + 1
        for chunk in chunks:
            chunk.metadata['source'] = filename
            chunk.metadata['doc_id'] = doc_id

        if global_vector_store is None:
            print(f"Creating new FAISS index for {len(chunks)} chunks...")
            global_vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            print(f"Adding {len(chunks)} new chunks to existing FAISS index...")
            global_vector_store.add_documents(chunks)
        
        global_vector_store.save_local(VECTOR_DB_PATH)

        DOCUMENTS_DB[doc_id] = {'filename': filename, 'content': doc_content, 'filepath': filepath}
        
        return jsonify({
            "message": "Document processed successfully!",
            "details": {
                "document_id": doc_id,
                "filename": filename,
                "file_extension": file_extension,
                "chunk_count": len(chunks),
                "status": "Ready for querying"
            }
        }), 200
    except requests.exceptions.RequestException as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Error downloading file from URL: {str(e)}"}), 500
    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error during document processing: {str(e)}")
        return jsonify({"error": f"Error processing document: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def process_query_api():
    """
    API endpoint to handle natural language queries and return a structured decision.
    This endpoint has been updated to use a more efficient and precise multi-step RAG chain.
    """
    data = request.json
    query = data.get('query')
    doc_id = data.get('doc_id')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if global_vector_store is None:
        return jsonify({"error": "No documents uploaded or vector store not initialized."}), 400

    retriever = global_vector_store.as_retriever(search_kwargs={"k": 5})

    # --- New Multi-Step RAG Logic for Precision and Efficiency ---
    
    # Step 1: Query Decomposition
    # Break down the user's query into smaller, targeted sub-queries.
    decomposition_prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Your task is to decompose a user's query into a set of multiple, simpler questions that can be answered by searching a document. Respond with a comma-separated list of questions.\n\nQuery: {input}\nDecomposed Questions:"
    )
    decomposer_chain = decomposition_prompt | llm | StrOutputParser()

    # Step 2: Parallel Retrieval and Fact Extraction
    # The function `retrieve_and_extract_facts` is a helper function that needs to be defined within the scope of this function or globally.
    # The lambda function in the original parallel chain was complex, so we'll simplify it.
    
    def retrieve_and_extract_facts(question):
        docs = retriever.get_relevant_documents(question)
        docs_str = "\n\n".join([doc.page_content for doc in docs])
        retrieval_and_fact_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following document snippets, extract only the specific fact that answers the question. If the document snippets do not contain the answer, state 'Not found'.\n\nQuestion: {question}\nContext: {context}"),
            ("user", "Question: {question}")
        ])
        chain = retrieval_and_fact_extraction_prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": docs_str})

    # Step 3: Final Decision Synthesis
    final_synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert policy analysis AI. Your task is to make a precise decision based on the user's query and the extracted facts from a document.

        Query: {query}
        Extracted Facts: {extracted_facts}
        Relevant Clauses: {relevant_clauses}
        
        Based on this information, provide a final decision and justification. If information is missing from the facts, state it clearly.
        
        Output your response as a JSON object strictly following this schema:
        {json_schema}
        """),
        ("user", "Query: {query}")
    ])
    
    # We will invoke the steps sequentially in the try block for clearer error handling
    output_parser = JsonOutputParser(pydantic_object=DecisionOutput)
    json_schema = output_parser.get_format_instructions()
    
    try:
        # Step 1: Decompose the original query
        decomposed_questions_list = decomposer_chain.invoke({"input": query})
        decomposed_questions = decomposed_questions_list.split(',')

        # Step 2: Extract facts for each question
        extracted_facts = [retrieve_and_extract_facts(q.strip()) for q in decomposed_questions]

        # Step 3: Get relevant clauses for the final synthesis
        relevant_clauses = retriever.get_relevant_documents(query)
        relevant_clauses_info = []
        for doc in relevant_clauses:
            relevant_clauses_info.append({
                "source_filename": doc.metadata.get('source', 'Unknown'),
                "content_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        # Step 4: Synthesize the final decision
        final_input = {
            "query": query,
            "extracted_facts": "\n".join(extracted_facts),
            "relevant_clauses": "\n\n".join([doc.page_content for doc in relevant_clauses]),
            "json_schema": json_schema,
        }
        synthesis_chain = final_synthesis_prompt | llm | output_parser
        result = synthesis_chain.invoke(final_input)

        # Build the final response object
        parsed_response = DecisionOutput(**result)
        parsed_response.RelevantClauses = relevant_clauses_info
        
        return jsonify(parsed_response.dict()), 200

    except Exception as e:
        print(f"Error during query processing: {e}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
