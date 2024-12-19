from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import Document
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "ibm-granite/granite-3.0-8b-instruct"
MODEL_CACHE_DIR = "/mnt/StorageOne/knowledgeGraph/transformers_cache"
os.environ["HF_HOME"] = MODEL_CACHE_DIR

# Check if the model is already downloaded
model_dir = Path(MODEL_CACHE_DIR) / f"models--{MODEL_NAME.replace('/', '--')}"
if not model_dir.exists():
    print(f"Model not found in {model_dir}. Downloading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
else:
    print(f"Model found in {model_dir}. Loading from cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Hugging Face pipeline for the language model
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    max_new_tokens=100  # Adjust based on desired generation length
)

# Wrap Hugging Face pipeline with HuggingFacePipeline from LangChain
hf_pipeline = HuggingFacePipeline(pipeline=chat_pipeline)

# Set up LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=hf_pipeline, strict_mode=False)

# Example text
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

# Create a Document object
documents = [Document(page_content=text)]

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,  # Reduced chunk size
    chunk_overlap=50,
    length_function=len,
)

# Split the documents
split_documents = text_splitter.split_documents(documents)

# Process each chunk and collect results
all_nodes = []
all_relationships = []

for doc in split_documents:
    graph_documents = llm_transformer.convert_to_graph_documents([doc])
    if graph_documents:
        all_nodes.extend(graph_documents[0].nodes)
        all_relationships.extend(graph_documents[0].relationships)

# Print results
print(f"Nodes: {all_nodes}")
print(f"Relationships: {all_relationships}")
