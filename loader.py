import os
import requests

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.graphs import Neo4jGraph

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

os.environ["NEO4J_URL"] = url

# embeddings = OllamaEmbeddings()
embeddings = OpenAIEmbeddings()

neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

neo4j_graph.query("CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE")
neo4j_graph.query("CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE")
neo4j_graph.query("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE")
neo4j_graph.query("CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE")

def load_so_data(tag: str = "neo4j", page: int = 1):
    base_url = "https://api.stackexchange.com/2.2/questions"
    parameters = (
        f"?pagesize=100&page={page}&order=desc&sort=creation&tagged={tag}"
        "&site=stackoverflow&filter=!6WPIomnMNcVD9"
    )
    data = requests.get(base_url + parameters).json()
    # Convert html to text and calculate embedding values
    for q in data["items"]:
        question_text = BeautifulSoup(q["body"], features="html.parser").text
        q["body"] = question_text
        q["embedding"] = embeddings.embed_query(q["title"] + " " + question_text)
        if q.get("answers"):
            for a in q.get("answers"):
                a["body"] = BeautifulSoup(a["body"], features="html.parser").text

    import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link,
        question.favorite_count = q.favorite_count, question.creation_date = q.creation_date,
        question.body = q.body, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = a.creation_date,
            answer.body = a.body
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """
    neo4j_graph.query(import_query, {"data": data["items"]})


dimension = 1536 # OpenAi
# dimension =  3xx # Ollama

def create_vector_index():
    # TODO use Neo4jVector Code from LangChain on the existing graph
    index_query = "CALL db.index.vector.createNodeIndex('stackoverflow', 'Question', 'embedding', dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query)
    except:  # Already exists
        pass


if __name__ == "__main__":
    create_vector_index()
    load_so_data("neo4j", 1)
