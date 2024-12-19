import os
import json
import faiss
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyPDF2 import PdfReader

# Helper function to print with timestamp
def tprint(message):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

# Set custom cache directory for Hugging Face models
custom_cache_path = "/mnt/StorageOne/knowledgeGraph/transformers_cache"
os.environ["HF_HOME"] = custom_cache_path
tprint(f"HF_HOME is set to: {custom_cache_path}")

# Paths and configurations
data_path = "./data"
index_path = "./faiss_index"
base_model_path = "ibm-granite/granite-3.0-8b-instruct"
document_mapping_path = "./document_mapping.txt"

# Distance threshold for relevance
DISTANCE_THRESHOLD = 0.5

def is_model_preinstalled(model_path):
    model_cache_path = os.path.join(custom_cache_path, "models--" + model_path.replace("/", "--"))
    tprint(f"Checking if model is pre-installed at: {model_cache_path}")
    return os.path.exists(model_cache_path)

# Load tokenizer and model
if 'tokenizer' not in globals() or 'model' not in globals():
    if is_model_preinstalled(base_model_path):
        tprint("Model and tokenizer found locally. Loading...")
    else:
        tprint("Model not found locally. Attempting to download...")
    tprint(f"Model will be downloaded to: {custom_cache_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=custom_cache_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=custom_cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tprint(f"Model loaded and moved to device: {device}")

# Determine embedding dimension from model config
d = model.config.hidden_size
tprint(f"Embedding dimension set to {d}")

# Load or create FAISS index
if os.path.exists(index_path):
    tprint(f"Loading existing FAISS index from {index_path}")
    index = faiss.read_index(index_path)
else:
    tprint("Creating a new FAISS index")
    index = faiss.IndexFlatL2(d)

def load_document_mapping():
    if os.path.exists(document_mapping_path):
        with open(document_mapping_path, 'r') as f:
            mapping = {int(line.split(' ')[0]): line.split(' ')[1].strip() for line in f}
        tprint("Loaded document mapping with entries:")
        for idx, path in mapping.items():
            tprint(f"  Doc ID: {idx}, Path: {path}")
        return mapping
    else:
        tprint("Document mapping file not found. Creating a new one.")
        return {}

def save_document_mapping(mapping):
    with open(document_mapping_path, 'w') as f:
        for idx, path in mapping.items():
            f.write(f"{idx} {path}\n")
    tprint("Document mapping saved.")

document_mapping = load_document_mapping()

def extract_text_from_pdf(file_path):
    tprint(f"Extracting text from PDF: {file_path}")
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        tprint(f"Error reading PDF {file_path}: {e}")
        return ""

def get_query_embedding(query_text):
    tprint(f"Generating embedding for query: {query_text}")
    inputs = tokenizer(query_text, return_tensors='pt').to(device)
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(inputs.input_ids)
    query_vector = input_embeds.mean(dim=1).squeeze(0).detach().cpu().numpy()
    tprint(f"Query embedding generated with shape {query_vector.shape}")
    return query_vector

def get_relevant_documents(query_text):
    tprint(f"Searching for relevant documents for query: {query_text}")

    # Check if the query explicitly mentions a file name
    if "name \"" in query_text and query_text.endswith(".pdf\""):
        # Extract the file name from the query
        file_name = query_text.split("name \"")[-1].split("\"")[0]
        tprint(f"File name explicitly mentioned: {file_name}")

        # Search for the file name in the document mapping
        for doc_id, doc_path in document_mapping.items():
            if file_name in doc_path:
                tprint(f"Exact file match found: {doc_path}")
                try:
                    embedding_vector = get_query_embedding(query_text)
                    distances, indices = index.search(embedding_vector.reshape(1, -1), k=5)

                    # Filter results to include only the specific file
                    relevant_texts = []
                    for i, doc_idx in enumerate(indices[0]):
                        if doc_idx == doc_id:
                            dist = distances[0][i]
                            tprint(f"Document {doc_idx} ({doc_path}) is relevant with distance {dist:.4f}")
                            if doc_path.endswith('.pdf'):
                                retrieved_text = extract_text_from_pdf(doc_path)
                            else:
                                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as doc_file:
                                    retrieved_text = doc_file.read()
                            relevant_texts.append(retrieved_text[:2000])
                    return relevant_texts

                except Exception as e:
                    tprint(f"Error processing file {doc_path}: {e}")
                    return []

        tprint(f"No exact file match found for {file_name}.")
        return []  # If no exact match is found, return an empty list

    else:
        # Fallback to regular FAISS search if no file name is specified
        query_vector = get_query_embedding(query_text)
        if index.ntotal > 0:
            distances, retrieved_docs = index.search(query_vector.reshape(1, -1), k=5)
            tprint(f"Retrieved documents with distances: {distances[0]}")
        else:
            tprint("Index is empty, no documents available for retrieval.")
            return []

        relevant_texts = []
        for i, doc_idx in enumerate(retrieved_docs[0]):
            dist = distances[0][i]
            if doc_idx != -1 and doc_idx in document_mapping and dist < DISTANCE_THRESHOLD:
                doc_path = document_mapping[doc_idx]
                tprint(f"Document {doc_idx} ({doc_path}) is relevant with distance {dist:.4f}")
                try:
                    if doc_path.endswith('.pdf'):
                        retrieved_text = extract_text_from_pdf(doc_path)
                    else:
                        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as doc_file:
                            retrieved_text = doc_file.read()
                    relevant_texts.append(retrieved_text[:2000])
                except Exception as e:
                    tprint(f"Error reading file {doc_path}: {e}")
        return relevant_texts

def main():
    tprint("RAG Pipeline LLM Ready. Type 'exit' to quit.")
    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() == 'exit':
            break
        
        retrieved_texts = get_relevant_documents(user_query)
        
        instruction = (
            "Below is some reference information related to your question. Only use it if it's relevant.\n"
            "DO NOT repeat the reference text or instructions verbatim.\n"
            "Use it ONLY to answer the user's question concisely.\n\n"
            "===REFERENCE START===\n"
        )
        reference_text = "\n".join(retrieved_texts) if retrieved_texts else "No relevant reference found."
        user_part = (
            "\n===REFERENCE END===\n"
            "User's question:\n"
            f"{user_query}\n\n"
            "Final Answer:"
        )
        context = instruction + reference_text + user_part

        response = ""
        try:
            inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=2048).to(device)
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            tprint(f"Error during response generation: {e}")
            response = "An error occurred while generating a response. Please try again."

        final_answer = response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response.strip()
        tprint("[FINAL OUTPUT]: " + final_answer)

if __name__ == "__main__":
    main()
