# tools/ingestion.py
import os
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import asyncio
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request as GoogleRequest

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 300
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load sentence embedding model
model = SentenceTransformer(MODEL_NAME)

# Global state for ingested data
doc_chunks, doc_embeddings = [], []

class DataIngestionTool:
    def __init__(self):
        self.name = "data_ingestion"
        self.description = "Ingest data from various sources (PDF, websites, YouTube, text) for RAG."
        
    def _get_google_credentials(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleRequest())
            else:
                client_config = {
                    "web": {
                        "client_id": os.getenv("GOOGLE_DRIVE_CLIENT_ID"),
                        "client_secret": os.getenv("GOOGLE_DRIVE_CLIENT_SECRET"),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, ['https://www.googleapis.com/auth/drive.readonly'])
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    def split_text(self, text, size=CHUNK_SIZE):
        """Splits text into fixed-size word chunks."""
        words = text.split()
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    def ingest_from_pdf(self, file_obj):
        """Processes the uploaded PDF and caches its embeddings."""
        global doc_chunks, doc_embeddings
        try:
            text = self.extract_pdf_text(file_obj)
            doc_chunks = self.split_text(text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return f"✅ Processed {len(doc_chunks)} chunks from PDF."
        except Exception as e:
            return f"❌ Failed to process PDF: {e}"

    def extract_pdf_text(self, file_obj):
        """Extracts and joins text from all pages of a PDF."""
        reader = PdfReader(file_obj)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def ingest_from_gdrive(self, file_type):
        """Ingests content from Google Drive (Docs/Slides)."""
        global doc_chunks, doc_embeddings
        try:
            creds = self._get_google_credentials()
            drive_service = build('drive', 'v3', credentials=creds)
            
            if file_type == "docs":
                mime_type = 'application/vnd.google-apps.document'
                export_mime = 'text/plain'
            elif file_type == "slides":
                mime_type = 'application/vnd.google-apps.presentation'
                export_mime = 'text/plain'
            else:
                return "Unsupported Google file type."
            
            results = drive_service.files().list(q=f"mimeType='{mime_type}'", pageSize=10, fields="files(id, name)").execute()
            items = results.get('files', [])
            
            if not items:
                return f"No Google {file_type.capitalize()} found in your Drive."
                
            full_text = ""
            for item in items:
                file_id = item['id']
                file_name = item['name']
                
                request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                fh.seek(0)
                full_text += fh.read().decode('utf-8') + "\n\n"
            
            doc_chunks = self.split_text(full_text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return f"✅ Successfully ingested {len(items)} Google {file_type.capitalize()}."

        except HttpError as e:
            return f"❌ Google API error: {e}"
        except Exception as e:
            return f"❌ Failed to ingest from Google Drive: {str(e)}"

    def ingest_from_website(self, url):
        """Ingests content from a website."""
        global doc_chunks, doc_embeddings
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            doc_chunks = self.split_text(text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return f"✅ Successfully ingested website from {url}. {len(doc_chunks)} chunks created."
        except Exception as e:
            return f"❌ Error ingesting website from {url}: {str(e)}"

    def ingest_from_youtube(self, url):
        """Ingests a YouTube video transcript."""
        global doc_chunks, doc_embeddings
        try:
            video_id = url.split("v=")[-1]
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([d['text'] for d in transcript_list])
            doc_chunks = self.split_text(text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return f"✅ Successfully ingested YouTube transcript. {len(doc_chunks)} chunks created."
        except Exception as e:
            return f"❌ Error ingesting YouTube transcript: {str(e)}"
    
    def ingest_from_text(self, text):
        """Ingests plain text."""
        global doc_chunks, doc_embeddings
        try:
            doc_chunks = self.split_text(text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return f"✅ Successfully ingested pasted text. {len(doc_chunks)} chunks created."
        except Exception as e:
            return f"❌ Error ingesting text: {str(e)}"
    
    def get_top_chunks(self, query, k=3):
        """Finds top-k relevant chunks using cosine similarity."""
        global doc_chunks, doc_embeddings
        if not doc_chunks or not isinstance(doc_embeddings, np.ndarray) or doc_embeddings.size == 0:
            return None
        try:
            query_emb = model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(query_emb, doc_embeddings)[0]
            indices = np.argsort(sims)[::-1][:k]
            return "\n\n".join([doc_chunks[i] for i in indices if i < len(doc_chunks)])
        except Exception as e:
            logger.error(f"Error in get_top_chunks: {e}")
            return None