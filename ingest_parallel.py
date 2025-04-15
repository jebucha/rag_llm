#!/usr/bin/env python3
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer

def process_pdf_file(file_path: str):
    """
    Process a single PDF file by extracting its text,
    splitting it into meaningful chunks, and returning
    the file path and list of chunks.

    :param file_path: Full path to the PDF file.
    :return: Tuple of (file_path, list of text chunks).
    """
    try:
        with open(file_path, 'rb') as f:
            # Open PDF file using fitz
            doc = fitz.open(stream=f.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Could not process '{file_path}': {e}")

    # Use a simple heuristic to split text: split on two or more newlines
    chunks = [chunk.strip() for chunk in re.split(r'\n{2,}', text) if len(chunk.strip()) > 40]
    return file_path, chunks

def main():
    pdf_dir = input("Please enter the path to your PDF files: ").strip()
    if not os.path.isdir(pdf_dir):
        print(f"Error: Directory '{pdf_dir}' does not exist. Exiting.")
        return

    # List all PDF files in the given directory.
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    if total_files == 0:
        print("No PDF files found in the directory.")
        return

    print(f"Found {total_files} PDF files. Starting ingestion...\n")
    
    results = []
    completed_files = 0

    # Use ProcessPoolExecutor to process files concurrently.
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_pdf_file, file): file for file in pdf_files}
        for future in as_completed(future_to_file):
            file_in_process = future_to_file[future]
            try:
                file_path, chunks = future.result()
                results.append((file_path, chunks))
            except Exception as error:
                print(f"Error processing '{file_in_process}': {error}")
            completed_files += 1
            percentage = (completed_files / total_files) * 100
            print(f"Processed {completed_files} of {total_files} files ({percentage:.2f}% complete)")
    
    # Aggregate all text chunks from all files.
    all_chunks = []
    for file_path, chunks in results:
        all_chunks.extend(chunks)
    print(f"\nIngestion complete: {len(all_chunks)} text chunks obtained from {total_files} PDF files.")

    # Use an embedding model that is well-suited for retrieval tasks.
    # For query models with a LLaMA-like architecture (e.g., ALIENTELLIGENCE/contractanalyzerv2),
    # a recommended embedding model is:
    embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    print(f"\nLoading embedding model '{embedding_model_name}' ...")
    model = SentenceTransformer(embedding_model_name)

    print("Computing embeddings for all text chunks...")
    # Batch compute embeddings. You can enable a progress bar here.
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print("Embeddings computed successfully.")

    # At this point, the embeddings can be stored in a vector database (e.g., Chroma DB)
    # along with metadata mapping back to their source files, IDs, etc.
    print("\nPDF ingestion and embedding pipeline completed.")

if __name__ == '__main__':
    main()

