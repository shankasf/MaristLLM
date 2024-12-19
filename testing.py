import os
import json
import faiss
import torch
import datetime
import logging
from time import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

import pdfplumber  # Make sure this is installed: pip install pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Set custom cache directory for Hugging Face models
custom_cache_path = "/mnt/StorageOne/knowledgeGraph/transformers_cache"
os.environ["HF_HOME"] = custom_cache_path
logger.info(f"HF_HOME is set to: {custom_cache_path}")

# Paths and configurations
data_path = "./data"
index_path = "./faiss_index"
base_model_path = "ibm-granite/granite-3.0-8b-instruct"
document_mapping_path = "./document_mapping.txt"

# Distance threshold for relevance
DISTANCE_THRESHOLD = 0.4
CHUNK_SIZE = 512  # Token chunk size

# Load or initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = embedding_model.get_sentence_embedding_dimension()

# Load or create FAISS index
def load_or_initialize_faiss_index():
    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        logger.info("Creating a new FAISS index")
        index = faiss.IndexFlatIP(embedding_dimension)
    return index

index = load_or_initialize_faiss_index()

def save_faiss_index():
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index saved to {index_path}")

def load_document_mapping():
    if os.path.exists(document_mapping_path):
        with open(document_mapping_path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded document mapping with {len(mapping)} entries.")
        return mapping
    else:
        logger.info("Document mapping file not found. Creating a new one.")
        return {}

def save_document_mapping(mapping):
    with open(document_mapping_path, 'w') as f:
        json.dump(mapping, f)
    logger.info("Document mapping saved.")

document_mapping = load_document_mapping()

def preprocess_text(text):
    # Split text into chunks for better indexing
    words = text.split()
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    return chunks

def add_to_index(text_chunks, file_path, doc_id):
    embeddings = embedding_model.encode(text_chunks)
    index.add(embeddings)
    document_mapping[doc_id] = {"file_path": file_path, "chunks": len(text_chunks)}
    logger.info(f"Successfully added {len(text_chunks)} chunks for Doc ID: {doc_id}")

def attempt_file_read(file_path):
    """
    Attempt to read the file as text. 
    For JSON: extract from prompt/response fields.
    For PDF: use pdfplumber to extract text.
    For other files: read as text with errors ignored.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = json.load(file)
            content_pieces = []
            if isinstance(data, list):
                for item in data:
                    text = (item.get('prompt', '') + "\n" + item.get('response', '')).strip()
                    if text:
                        content_pieces.append(text)
            elif isinstance(data, dict):
                text = (data.get('prompt', '') + "\n" + data.get('response', '')).strip()
                if text:
                    content_pieces.append(text)
            return content_pieces

    elif file_path.endswith('.pdf'):
        # Extract text from PDF using pdfplumber
        content_pieces = []
        try:
            with pdfplumber.open(file_path) as pdf:
                all_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)
                if all_text:
                    # Combine all pages into a single string
                    combined_text = "\n".join(all_text).strip()
                    if combined_text:
                        content_pieces.append(combined_text)
            return content_pieces
        except Exception as e:
            logger.warning(f"Could not extract text from PDF {file_path}: {e}")
            return []

    else:
        # For all other file types, try reading as text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read().strip()
                if text:
                    return [text]
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
        return []

def populate_index(data_folder):
    logger.info("Populating FAISS index with local file embeddings...")
    next_doc_id = len(document_mapping)

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)

        # Skip already indexed files
        if any(file_path == entry["file_path"] for entry in document_mapping.values()):
            continue

        # Attempt to read content from the file
        content_list = attempt_file_read(file_path)
        if not content_list:
            logger.info(f"No content extracted from {file_name}, skipping.")
            continue

        # Index all extracted content pieces
        for content in content_list:
            text_chunks = preprocess_text(content)
            add_to_index(text_chunks, file_path, next_doc_id)
            next_doc_id += 1

    save_faiss_index()
    save_document_mapping(document_mapping)

def search_index(query):
    query_embedding = embedding_model.encode([query])[0]
    start_time = time()
    distances, indices = index.search(query_embedding.reshape(1, -1), k=5)
    end_time = time()
    logger.info(f"Search completed in {end_time - start_time:.4f} seconds. Retrieved indices: {indices}, distances: {distances}")
    return distances, indices

def retrieve_relevant_documents(indices, distances):
    relevant_texts = []
    for i, doc_idx in enumerate(indices[0]):
        if doc_idx == -1 or distances[0][i] >= DISTANCE_THRESHOLD:
            continue
        doc_info = document_mapping.get(doc_idx)
        if doc_info and os.path.exists(doc_info["file_path"]):
            # Read a portion of the file as context
            try:
                with open(doc_info["file_path"], 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                relevant_texts.append(text[:2000])  # Limiting the text length for context
            except Exception as e:
                logger.warning(f"Could not re-read file {doc_info['file_path']} for context: {e}")
    return relevant_texts

def generate_response(query, context):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=custom_cache_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=custom_cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    instruction = (
        "Below is some reference information related to your question. Only use it if it's relevant.\n"
        "DO NOT repeat the reference text or instructions verbatim.\n"
        "Use it ONLY to answer the user's question concisely.\n\n"
        "===REFERENCE START===\n"
    )

    reference_text = "\n".join(context) if context else "No relevant documents found."
    user_part = (
        "\n===REFERENCE END===\n"
        "User's question:\n"
        f"{query}\n\n"
        "Final Answer:"
    )

    input_text = instruction + reference_text + user_part
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    populate_index(data_path)
    logger.info("RAG Pipeline Initialized. Type 'exit' to quit.")
    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() == 'exit':
            break

        distances, indices = search_index(user_query)
        relevant_documents = retrieve_relevant_documents(indices, distances)
        final_answer = generate_response(user_query, relevant_documents)
        logger.info(f"[FINAL OUTPUT]: {final_answer}")

if __name__ == "__main__":
    main()
