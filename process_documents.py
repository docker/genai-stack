import os
import cv2
import json
import requests
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from neo4j import GraphDatabase
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
STACKAI_LLM_API = "http://localhost:8000/v1/completions"

# Initialize Neo4j connection
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load TrOCR for OCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess_image(image_path):
    """Preprocess image for OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return image

def extract_text_from_image(image_path):
    """Extract handwritten text from image using TrOCR."""
    image = preprocess_image(image_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfminer."""
    return extract_pdf_text(pdf_path)

def extract_text_from_docx(docx_path):
    """Extract text from DOCX using python-docx."""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path):
    """Extract text from TXT file."""
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_math(image_path):
    """Extract mathematical formulas."""
    response = requests.post("https://huggingface.co/breezedeus/pix2text-mfr", 
                             files={"file": open(image_path, "rb")})
    return response.json().get("text", "")

def extract_chemistry(image_path):
    """Extract chemical formulas."""
    response = requests.post("https://chemocr.ai/api/v1/predict", 
                             files={"file": open(image_path, "rb")})
    return response.json().get("chemical_formula", "")

def summarize_content(text):
    """Summarize extracted content using LLM."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize this scientific note and extract key concepts: {text}"
    )
    chain = LLMChain(llm=OpenAI(base_url=STACKAI_LLM_API, api_key="your-api-key"), prompt=prompt)
    return chain.run(text)

def store_in_neo4j(filename, text, math, chem, summary):
    """Store extracted information in Neo4j as a knowledge graph."""
    with driver.session() as session:
        session.run("""
            CREATE (n:Document {filename: $filename, text: $text, summary: $summary})
        """, filename=filename, text=text, summary=summary)

        for formula in math.split("\n"):
            if formula:
                session.run("""
                    MATCH (n:Document {filename: $filename})
                    CREATE (m:MathFormula {formula: $formula})-[:APPEARS_IN]->(n)
                """, filename=filename, formula=formula)

        for compound in chem.split("\n"):
            if compound:
                session.run("""
                    MATCH (n:Document {filename: $filename})
                    CREATE (c:Chemical {compound: $compound})-[:APPEARS_IN]->(n)
                """, filename=filename, compound=compound)

def process_files(directory):
    """Recursively process files in a directory and store them in Neo4j."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing: {file_path}")

            extracted_text = ""
            extracted_math = ""
            extracted_chem = ""

            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                extracted_text = extract_text_from_image(file_path)
                extracted_math = extract_math(file_path)
                extracted_chem = extract_chemistry(file_path)
            elif file.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif file.lower().endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)
            elif file.lower().endswith(".txt"):
                extracted_text = extract_text_from_txt(file_path)

            full_content = f"{extracted_text}\nMath: {extracted_math}\nChem: {extracted_chem}"
            summary = summarize_content(full_content)

            store_in_neo4j(file, extracted_text, extracted_math, extracted_chem, summary)

def generate_knowledge_graph():
    """Generate a visualization of the knowledge graph."""
    query = "MATCH (n)-[r]->(m) RETURN n, r, m"
    nodes = []
    relationships = []

    with driver.session() as session:
        results = session.run(query)
        for record in results:
            n, r, m = record["n"], record["r"], record["m"]
            nodes.append(n["filename"] if "filename" in n else n["formula"] if "formula" in n else n["compound"])
            nodes.append(m["filename"] if "filename" in m else m["formula"] if "formula" in m else m["compound"])
            relationships.append((n["filename"] if "filename" in n else n["formula"] if "formula" in n else n["compound"], 
                                  m["filename"] if "filename" in m else m["formula"] if "formula" in m else m["compound"]))

    G = nx.Graph()
    G.add_edges_from(relationships)

    plt.figure(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    plt.title("Knowledge Graph")
    plt.savefig("knowledge_graph.png")
    print("Knowledge graph saved as knowledge_graph.png")

if __name__ == "__main__":
    dir_path = input("Enter the directory path to process: ")
    process_files(dir_path)
    generate_knowledge_graph()