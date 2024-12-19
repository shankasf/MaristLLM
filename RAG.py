import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import re
import logging
from pathlib import Path
from collections import Counter
import os
import glob
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphRAG:
    def __init__(self, text_file_path, neo4j_uri, neo4j_user, neo4j_password, model_name, model_cache_dir):
        self.text_file_path = text_file_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimensions = 384
        self.chat_model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.chat_tokenizer, self.chat_model = self.load_model_from_cache_or_download(self.chat_model_name, self.model_cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat_model.to(self.device)
        self.kg = Neo4jGraph(url=self.neo4j_uri, username=self.neo4j_user, password=self.neo4j_password)
        print(f"DEBUG: Initialized KnowledgeGraphRAG with device: {self.device}")

    def load_model_from_cache_or_download(self, model_name, cache_dir):
        model_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
        if not model_dir.exists():
            print(f"DEBUG: Model not found in {model_dir}. Downloading...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            print(f"DEBUG: Model found in {model_dir}. Loading from cache...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        return tokenizer, model

    def process_folder(self, folder_path):
        print(f"DEBUG: Processing folder: {folder_path}")
        for file_path in glob.glob(os.path.join(folder_path, '*')):
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()

            # Check if the file is already processed
            if '_processed' in file_name:
                print(f"DEBUG: File already processed: {file_name}")
                continue

            print(f"DEBUG: Processing file: {file_name}")
            if file_extension == '.pdf':
                text = self.pdf_to_text(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:
                continue

            self.text_file_path = file_path
            self.create_unstructured_kg(text)
            self.create_structured_kg(text)

            # Append '_processed' before the file extension
            new_file_path = f"{os.path.splitext(file_path)[0]}_processed{file_extension}"
            os.rename(file_path, new_file_path)
            print(f"DEBUG: Processed and renamed: {new_file_path}")


    def pdf_to_text(self, pdf_path):
        print(f"DEBUG: Converting PDF to text: {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def split_text_into_chunks(self, text, chunk_size=2000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        print(f"DEBUG: Split text into {len(chunks)} chunks")
        return [{"text": chunk, "chunkId": f"chunk-{i:04d}", "chunkSeqId": i} for i, chunk in enumerate(chunks)]

    def create_unstructured_kg(self, text):
        print("DEBUG: Creating unstructured knowledge graph")
        self.kg.query("CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE")
        chunks = self.split_text_into_chunks(text)
        self.kg.query(f"""
        CREATE VECTOR INDEX data_chunks_index IF NOT EXISTS FOR (c:Chunk)
        ON (c.textEmbedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """)
        for chunk in chunks:
            self.kg.query("""
            MERGE(c:Chunk {chunkId: $chunkParam.chunkId})
            ON CREATE SET c.chunkSeqId = $chunkParam.chunkSeqId, c.text = $chunkParam.text
            """, params={'chunkParam': chunk})
            chunk_embedding = self.embeddings.embed_query(chunk['text'])
            self.kg.query("""
            MATCH (c:Chunk {chunkId: $chunkId})
            CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
            RETURN c
            """, params={'chunkId': chunk['chunkId'], 'embedding': chunk_embedding})
        print("DEBUG: Unstructured knowledge graph created")

    def create_structured_kg(self, text):
        print("DEBUG: Creating structured knowledge graph")
        chunks = self.split_text_into_chunks(text)
        for chunk in chunks:
            entities, relations = self.extract_entities_relations(chunk['text'])
            print(f"DEBUG: Extracted entities: {entities}")
            print(f"DEBUG: Extracted relations: {relations}")
            for entity in entities:
                self.kg.query("""
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.type = $type
                """, params=entity)
            for relation in relations:
                self.kg.query("""
                MATCH (s:Entity {id: $from}), (o:Entity {id: $to})
                CALL apoc.create.relationship(s, $type, {}, o)
                YIELD rel
                RETURN rel
                """, params=relation)
        print("DEBUG: Structured knowledge graph created")

    def extract_entities_relations(self, text):
        print("DEBUG: Extracting entities and relations")
        prompt = self.create_extraction_prompt(text)
        inputs = self.chat_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        outputs = self.chat_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
        response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_extraction_output(response)

    def create_entity_index(self):
        print("DEBUG: Creating entity index")
        self.kg.query("""
        CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)
        """)
        print("DEBUG: Index 'entity_id_index' created")

    def create_extraction_prompt(self, text):
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

    def parse_extraction_output(self, response):
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
        print(f"DEBUG: Parsed {len(entities)} entities and {len(relations)} relations")
        return entities, relations

    def vector_search(self, question, top_k=5):
        print(f"DEBUG: Performing vector search for question: {question}")
        question_embedding = self.embeddings.embed_query(question)
        results = self.kg.query("""
        CALL db.index.vector.queryNodes('data_chunks_index', $top_k, $question_embedding)
        YIELD node, score
        RETURN score, node.text AS text
        """, params={'top_k': top_k, 'question_embedding': question_embedding})
        return [{"score": record["score"], "text": record["text"]} for record in results]

    def entity_search(self, entities):
        print(f"DEBUG: Performing entity search for entities: {entities}")
        results = []
        for entity in entities:
            query_result = self.kg.query("""
            MATCH (e:Entity {id: $entity})-[r]-(related)
            RETURN e, r, related
            """, params={'entity': entity})
            results.extend(query_result)
        return results

    def structured_retriever(self, question: str) -> str:
        print(f"DEBUG: Structured retrieval for question: {question}")
        result = ""
        entities, _ = self.extract_entities_relations(question)
        for entity in entities:
            print(f"DEBUG: Getting Entity: {entity}")
            response = self.kg.query(
                """
                MATCH (e:Entity)
                WHERE e.id CONTAINS $query
                WITH e LIMIT 2
                CALL {
                    WITH e
                    MATCH (e)-[r]->(neighbor)
                    RETURN e.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    MATCH (e)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + e.id AS output
                }
                RETURN output
                LIMIT 20
                """,
                {"query": entity['id']}
            )
            result += "\n".join([el["output"] for el in response])
        return result

    def answer_question(self, question):
        print(f"DEBUG: Answering question: {question}")
        DISTANCE_THRESHOLD = 0.5
        vector_results = self.vector_search(question)
        print(f"DEBUG: Vector Results (Length: {len(vector_results)}): {vector_results}")
        print(f"DEBUG: Vector Search Results: {vector_results}")
        if isinstance(vector_results, list) and all(isinstance(item, dict) for item in vector_results):
            retrieved_texts = [result["text"][:2000] for result in vector_results[:min(len(vector_results), 1)]]
            print(f"DEBUG: Retrieved Texts (Length: {len(retrieved_texts)}): {retrieved_texts}")
        else:
            print("DEBUG: Invalid vector_results format.")
        structured_results = self.structured_retriever(question)
        print(f"DEBUG: Structured Retriever Results: {structured_results}")
        instruction = (
            "Below is some reference information related to your question. Only use it if it's relevant.\n"
            "DO NOT repeat the reference text or instructions verbatim.\n"
            "Use it ONLY to answer the user's question concisely.\n\n"
            "===REFERENCE START===\n"
        )
        unstructured_text = "\n".join(retrieved_texts) if retrieved_texts else ""
        structured_text = structured_results
        user_part = (
            "\n===REFERENCE END===\n"
            "User's question:\n"
            f"{question}\n\n"
            "Final Answer:"
        )
        context = instruction + unstructured_text + "\n" + structured_text + user_part
        print(f"DEBUG: Full context: {context}")
        print(f"DEBUG: Context length: {len(self.chat_tokenizer(context, return_tensors='pt').input_ids[0])}")
        if len(self.chat_tokenizer(context, return_tensors='pt').input_ids[0]) > 2048:
            print("DEBUG: Context is too long, splitting it.")
            instruction_part = context.split('===REFERENCE END===')[0]
            user_part = context.split('===REFERENCE END===')[1]
            inputs = self.chat_tokenizer(instruction_part + user_part, return_tensors='pt', truncation=True, max_length=2048).to(self.device)
        else:
            inputs = self.chat_tokenizer(context, return_tensors='pt', truncation=True, max_length=2048).to(self.device)
        outputs = self.chat_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512
        )
        response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"DEBUG: Generated Response: {response}")
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response.strip()
        return final_answer

# Usage example
if __name__ == "__main__":
    folder_path = "/mnt/StorageOne/knowledgeGraph/MaristLLM/data"
    neo4j_uri = "bolt://10.11.28.104:7687"
    neo4j_user = "neo4j"
    neo4j_password = "@lenovoserverneopswd"
    model_name = "ibm-granite/granite-3.0-8b-instruct"
    model_cache_dir = "/mnt/StorageOne/knowledgeGraph/transformers_cache"

    rag = KnowledgeGraphRAG(None, neo4j_uri, neo4j_user, neo4j_password, model_name, model_cache_dir)
    #rag.process_folder(folder_path)
    rag.create_entity_index()

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        answer = rag.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
