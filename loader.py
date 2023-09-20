import os
import requests

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.graphs import Neo4jGraph

import streamlit as st

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

os.environ["NEO4J_URL"] = url

# embeddings = OllamaEmbeddings(base_url=ollama_base_url)
# dimension =  4096 # Ollama

embeddings = OpenAIEmbeddings()
dimension = 1536  # OpenAi

neo4j_graph = Neo4jGraph(url=url, username=username, password=password)


def create_constraints():
    neo4j_graph.query(
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE"
    )
    neo4j_graph.query(
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE"
    )
    neo4j_graph.query(
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE"
    )
    neo4j_graph.query(
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE"
    )


create_constraints()

def create_vector_index(dimension):
    # TODO use Neo4jVector Code from LangChain on the existing graph
    index_query = "CALL db.index.vector.createNodeIndex('stackoverflow', 'Question', 'embedding', $dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


create_vector_index(dimension)


def load_so_data(tag: str = "neo4j", page: int = 1) -> None:
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


# Streamlit
def get_tag() -> str:
    input_text = st.text_input("Which tag questions do you want to import?", value="neo4j")
    return input_text


def get_pages():
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input("Number of pages (100 questions per page)", step=1, min_value=1)
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1)
    st.caption("Only questions with answers will be imported.")
    return (int(num_pages), int(start_page))

st.header("StackOverflow Loader")
st.subheader("Choose StackOverflow tags to load into Neo4j")
st.caption("Go to http://localhost:7474/browser/ to explore the graph.")

user_input = get_tag()
num_pages, start_page = get_pages()

if st.button("Import"):
    with st.spinner("Loading... This might take a minute or two."):
        try:
            for page in range(1, num_pages + 1):
                load_so_data(user_input, start_page + (page-1))
            st.success("Import successful", icon="✅")
        except Exception as e:
            st.error(f"Error: {e}", icon="🚨")
