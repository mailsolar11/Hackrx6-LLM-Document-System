import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = 'uploads'
# Added new extensions: 'csv', 'json', 'html'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'eml', 'csv', 'json', 'html'}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DOCUMENTS_DB = {}
VECTOR_DB_PATH = "faiss_index.bin"