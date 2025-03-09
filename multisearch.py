import os
import json
import cv2
import requests
import torch
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fastapi import FastAPI, UploadFile, File
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

# Initialize FastAPI
app = FastAPI()

# Load OCR Model (TrOCR)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def preprocess_image(image_path):
    """Preprocess image for OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return image

def extract_text(image_path):
    """Extract text from handwritten notes."""
    image = preprocess_image(image_path)
    image = cv2.resize(image, (1024, 1024))

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extracted_text

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
    """Summarize scientific notes using LLM."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize this scientific note and extract key concepts: {text}"
    )
    chain = LLMChain(llm=OpenAI(base_url=STACKAI_LLM_API, api_key="your-api-key"), prompt=prompt)
    return chain.run(text)

def store_in_neo4j(title, text, math, chem, summary):
    """Store extracted information in Neo4j as a knowledge graph."""
    with driver.session() as session:
        session.run("""
            CREATE (n:Note {title: $title, text: $text, summary: $summary})
        """, title=title, text=text, summary=summary)

        for formula in math.split("\n"):
            if formula:
                session.run("""
                    MATCH (n:Note {title: $title})
                    CREATE (m:MathFormula {formula: $formula})-[:APPEARS_IN]->(n)
                """, title=title, formula=formula)

        for compound in chem.split("\n"):
            if compound:
                session.run("""
                    MATCH (n:Note {title: $title})
                    CREATE (c:Chemical {compound: $compound})-[:APPEARS_IN]->(n)
                """, title=title, compound=compound)

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image and store in Neo4j."""
    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text(file_path)
    extracted_math = extract_math(file_path)
    extracted_chem = extract_chemistry(file_path)

    full_content = f"{extracted_text}\nMathematical Formulas: {extracted_math}\nChemical Formulas: {extracted_chem}"
    summary = summarize_content(full_content)

    store_in_neo4j(file.filename, extracted_text, extracted_math, extracted_chem, summary)

    return {"summary": summary, "math": extracted_math, "chem": extracted_chem}

@app.get("/knowledge-graph/")
def get_knowledge_graph():
    """Retrieve knowledge graph data from Neo4j and visualize it."""
    query = """
    MATCH (n)-[r]->(m) RETURN n, r, m
    """
    nodes = []
    relationships = []

    with driver.session() as session:
        results = session.run(query)
        for record in results:
            n, r, m = record["n"], record["r"], record["m"]
            nodes.append(n["title"] if "title" in n else n["formula"] if "formula" in n else n["compound"])
            nodes.append(m["title"] if "title" in m else m["formula"] if "formula" in m else m["compound"])
            relationships.append((n["title"] if "title" in n else n["formula"] if "formula" in n else n["compound"], 
                                  m["title"] if "title" in m else m["formula"] if "formula" in m else m["compound"]))

    G = nx.Graph()
    G.add_edges_from(relationships)

    plt.figure(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    plt.title("Knowledge Graph")
    plt.savefig("./temp/knowledge_graph.png")
    return {"message": "Knowledge graph updated. View the generated graph at /temp/knowledge_graph.png"}

if __name__ == "__main__":
    import uvicorn
    os.makedirs("./temp", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8080)