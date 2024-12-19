from dotenv import load_dotenv
import os
import re
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

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
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Debugging command: Print device information
print(f"Using device: {device}")



# Define a prompt to extract entities and relationships
def create_prompt(text):
    return f"""
<#meta#>
- Task: Extract entities and relationships
<#system#>
You are an AI assistant specialized in extracting structured information from text.
<#chat#>
<#user#>
{text}
<#bot#>
Please extract entities and their relationships in the following format:
Entities:
- entity_name: entity_type
Relationships:
- subject -> predicate -> object
"""

# Define text to process
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

# Generate the output from the model
prompt = create_prompt(text)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

# Debugging command: Print tokenized inputs
print(f"Tokenized inputs: {inputs}")

outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Debugging command: Print raw model response
print(f"Raw model response: {response}")

# Parse the output to extract entities and relationships

""" def parse_output(response):
    entities = []
    relations = []
    entity_pattern = r'- (.+): (.+)'
    relation_pattern = r'- (.+)'

    lines = response.split('\n')
    parsing_entities = False
    parsing_relations = False

    for line in lines:
        if line.strip() == 'Entities:':
            parsing_entities = True
            parsing_relations = False
        elif line.strip() == 'Relationships:':
            parsing_entities = False
            parsing_relations = True
        elif parsing_entities:
            match = re.match(entity_pattern, line.strip())
            if match:
                entity, entity_type = match.groups()
                entities.append((entity.strip(), entity_type.strip()))
        elif parsing_relations:
            match = re.match(relation_pattern, line.strip())
            if match:
                relation = match.group(1)
                # Attempt to split the relation into subject, predicate, and object
                parts = relation.split(' ', 2)
                if len(parts) == 3:
                    relations.append((parts[0], parts[1], parts[2]))
                else:
                    # If we can't split it properly, add it as is
                    relations.append((relation, '', ''))

    return entities, relations """


def parse_output(response):
    entities = []
    relations = []
    entity_pattern = r'- (.+): (.+)'
    relation_pattern = r'- (.+) -> (.+) -> (.+)'

    lines = response.split('\n')
    parsing_entities = False
    parsing_relations = False

    for line in lines:
        if line.strip() == 'Entities:':
            parsing_entities = True
            parsing_relations = False
        elif line.strip() == 'Relationships:':
            parsing_entities = False
            parsing_relations = True
        elif parsing_entities:
            match = re.match(entity_pattern, line.strip())
            if match:
                entity, entity_type = match.groups()
                entities.append({"id": entity.strip(), "type": entity_type.strip()})
        elif parsing_relations:
            match = re.match(relation_pattern, line.strip())
            if match:
                subject, predicate, object = match.groups()
                relations.append({"from": subject.strip(), "type": predicate.strip(), "to": object.strip()})

    return entities, relations

entities, relations = parse_output(response)

# Create graph_documents
graph_documents = [{
    "entities": entities,
    "relations": relations,
    "source": text  # This adds the original text as the source
}]

# Now you can use graph_documents with kg.add_graph_documents()
# res = kg.add_graph_documents(
#     graph_documents,
#     include_source=True,
#     baseEntityLabel=True,
# )

print("graph_documents for Neo4j:")
print(graph_documents)


""" entities, relations = parse_output(response)

# Debugging command: Print parsed entities and relationships
print(f"Parsed entities: {entities}")
print(f"Parsed relationships: {relations}")

# Display entities and relationships
print("Entities:")
for entity in entities:
    print(f"- {entity[0]} ({entity[1]})")

print("\nRelations:")
for relation in relations:
    print(f"- {relation[0]} -> {relation[1]} -> {relation[2]}")

# Build a knowledge graph (basic example)
def build_knowledge_graph(entities, relations):
    graph = defaultdict(list)
    entity_set = {entity[0] for entity in entities}  # Create a set of valid entities
    for sub, rel, obj in relations:
        if sub in entity_set and obj in entity_set:  # Only add valid relationships
            graph[sub].append((rel, obj))
        else:
            print(f"Skipping invalid relationship: ({sub}, {rel}, {obj})")
    return graph

knowledge_graph = build_knowledge_graph(entities, relations)

# Debugging command: Print the knowledge graph structure
print(f"Knowledge graph structure: {knowledge_graph}")

print("\nKnowledge Graph:")
for subject, edges in knowledge_graph.items():
    for edge in edges:
        print(f"- {subject} -> {edge[0]} -> {edge[1]}")
 """