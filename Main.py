import os
import json
import faiss
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def add_to_faiss_index(text, file_path, doc_id):
    tprint(f"Adding file: {file_path} to FAISS index with Doc ID: {doc_id}")
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(inputs.input_ids).mean(dim=1).squeeze(0).detach().cpu().numpy()
    index.add(input_embeds.reshape(1, -1))
    document_mapping[doc_id] = file_path
    tprint(f"Successfully added embedding for Doc ID: {doc_id}")

def populate_faiss_index(data_folder):
    tprint("Populating FAISS index with local file embeddings...")
    next_doc_id = len(document_mapping)
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_path in document_mapping.values():
            # Already indexed
            continue

        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                add_to_faiss_index(text, file_path, next_doc_id)
                next_doc_id += 1
        elif file_name.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    text = item.get('prompt', '') + "\n" + item.get('response', '')
                    add_to_faiss_index(text, file_path, next_doc_id)
                    next_doc_id += 1

    # Save the FAISS index and document mapping
    faiss.write_index(index, index_path)
    save_document_mapping(document_mapping)
    tprint(f"FAISS index saved to {index_path}")

def get_query_embedding(query_text):
    tprint(f"Generating embedding for query: {query_text}")
    inputs = tokenizer(query_text, return_tensors='pt').to(device)
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(inputs.input_ids)
    query_vector = input_embeds.mean(dim=1).squeeze(0).detach().cpu().numpy()
    tprint(f"Query embedding generated with shape {query_vector.shape}")
    return query_vector

def main():
    tprint("RAG Pipeline LLM Ready. Type 'exit' to quit.")
    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() == 'exit':
            break

        query_vector = get_query_embedding(user_query)
        if index.ntotal > 0:
            distances, retrieved_docs = index.search(query_vector.reshape(1, -1), k=5)
            tprint(f"Retrieved documents with distances: {distances}")
        else:
            retrieved_docs = [[-1]]
            tprint("Index is empty, no documents retrieved")

        retrieved_texts = []
        # Only add documents if they meet the distance threshold
        for i, doc_idx in enumerate(retrieved_docs[0]):
            dist = distances[0][i]
            if doc_idx != -1 and doc_idx in document_mapping and dist < DISTANCE_THRESHOLD:
                doc_path = document_mapping[doc_idx]
                if os.path.exists(doc_path):
                    tprint(f"Retrieving relevant content from Doc ID: {doc_idx}, Distance: {dist}, Path: {doc_path}")
                    with open(doc_path, 'r', encoding='utf-8') as doc_file:
                        retrieved_text = doc_file.read()
                        retrieved_texts.append(retrieved_text[:2000])
            else:
                # If distance is too large or doc not found, skip it
                pass

        # Instruction and prompt structure
        instruction = (
            "Below is some reference information related to your question. Only use it if it's relevant.\n"
            "DO NOT repeat the reference text or instructions verbatim.\n"
            "Use it ONLY to answer the user's question concisely.\n\n"
            "===REFERENCE START===\n"
        )

        reference_text = "\n".join(retrieved_texts) if retrieved_texts else ""
        user_part = (
            "\n===REFERENCE END===\n"
            "User's question:\n"
            f"{user_query}\n\n"
            "Final Answer:"
        )

        context = instruction + reference_text + user_part

        inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=2048).to(device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the response to remove instructions, context, and user query.
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response.strip()

        tprint("[FINAL OUTPUT]: " + final_answer)

if __name__ == "__main__":
    populate_faiss_index(data_path)
    main()

#Sagar