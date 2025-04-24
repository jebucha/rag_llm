#!/usr/bin/env python3

import streamlit as st
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import chromadb
from datetime import datetime
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.client = chromadb.Client()
        self.vector_store = None
        self.supported_formats = {'.pdf': self.extract_text_from_pdf,
                                 '.txt': self.extract_text_from_txt,
                                 '.docx': self.extract_text_from_docx}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    @staticmethod
    def extract_text_from_pdf(file):
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def extract_text_from_txt(file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading txt file: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file):
        from docx import Document
        document = Document(BytesIO(file.getvalue()))
        return '\n'.join([para.text for para in document.paragraphs])

    def process_file(self, file) -> Dict[str, Any]:
        if not self.supported_formats.get(os.path.splitext(file.name)[1].lower()):
            raise ValueError(f"Unsupported file format: {file.name}")

        try:
            text = self.supported_formats[os.path.splitext(file.name)[1]](file)
            metadata = DocumentMetadata(
                source=file.name,
                type=os.path.splitext(file.name)[1],
                created_at=datetime.now().isoformat(),
                author=st.session_state.get('user', 'anonymous')
            )

            chunks, embeddings = self._process_text(text)
            return {
                "text": text,
                "chunks": chunks,
                "embeddings": embeddings.tolist(),
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            raise

    def _process_text(self, text: str) -> tuple[List[str], List[float]]:
        # Implement chunking and embedding logic here
        chunks = self.chunk_text(text)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 20) -> List[str]:
        # Implement text chunking logic here
        pass

    @staticmethod
    def generate_embeddings(chunks: List[str]) -> Any:
        # Implement embedding generation logic here
        pass

class DocumentMetadata:
    def __init__(self, source: str, type: str, created_at: str = None, author: str = None):
        self.source = source
        self.type = type
        self.created_at = created_at or datetime.now().isoformat()
        self.author = author or "unknown"

def main():
    st.set_page_config(page_title="Document Processing System", layout="wide")

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = ""

    config_path = Path("config.json")
    processor = DocumentProcessor(config_path)

    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Ingest Documents", "Verify Collections"])

    with tab1:
        ask_questions(processor)

    with tab2:
        ingest_documents(processor)

    with tab3:
        verify_collections(processor)

def ask_questions(processor: DocumentProcessor):
    query = st.text_input("Enter your question:")

    if query and processor.vector_store:
        try:
            answer, source_docs = processor.get_answer(query)
            st.write(f"Answer: {answer}")

            with st.expander("Source Documents"):
                for doc in source_docs[:5]:  # Show top 5 documents
                    st.write(doc.metadata.source)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            st.error("An error occurred while processing your query.")

def ingest_documents(processor: DocumentProcessor):
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )

    if uploaded_files:
        with st.spinner('Processing files...'):
            results = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(processor.process_file, file) for file in uploaded_files]
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing file: {e}")

            if st.button("Save to Database"):
                try:
                    processor.save_to_database(results)
                    st.success("Files saved successfully!")
                except Exception as e:
                    logger.error(f"Error saving to database: {e}")
                    st.error("Failed to save files.")

def verify_collections(processor: DocumentProcessor):
    collections = processor.get_collections()

    if not collections:
        st.write("No collections found.")
        return

    selected_collection = st.selectbox(
        "Select a collection",
        options=collections
    )

    if selected_collection and st.button("Delete Collection"):
        try:
            processor.delete_collection(selected_collection)
            st.success(f"Collection '{selected_collection}' deleted successfully!")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            st.error(f"Failed to delete '{selected_collection}'.")

if __name__ == "__main__":
    main()
