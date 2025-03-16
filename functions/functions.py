from neo4j import GraphDatabase
from dotenv import load_dotenv
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

# Define your Neo4j connection details
##neo4j_uri = "bolt://192.168.1.153:7687"  # Replace with your Neo4j host and port
#neo4j_username = "neo4j"  # Replace with your Neo4j username
#neo4j_password = "password"  # Replace with your Neo4j password

# Connect to the Neo4j database
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

cypher_query = """
MATCH (n)
RETURN n
"""

def run_cypher_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]


# Run the Cypher query
#results = run_cypher_query(cypher_query)

# Process and print the results
#for record in results:
    ###print(record)

# Close the Neo4j driver
#driver.close()
#