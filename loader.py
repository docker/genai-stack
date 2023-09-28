import os
import requests
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from utils import load_embedding_model

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")

os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={ollama_base_url: ollama_base_url}, logger=logger
)

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
    index_query = "CALL db.index.vector.createNodeIndex('stackoverflow', 'Question', 'embedding', $dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass
    index_query = "CALL db.index.vector.createNodeIndex('top_answers', 'Answer', 'embedding', $dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


create_vector_index(dimension)


so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"


def load_so_data(tag: str = "neo4j", page: int = 1) -> None:
    parameters = (
        f"?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}"
        "&site=stackoverflow&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb"
    )
    data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)


def load_high_score_so_data() -> None:
    parameters = (
        f"?fromdate=1664150400&order=desc&sort=votes&site=stackoverflow&"
        "filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII"
    )
    data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)


def insert_so_data(data: dict) -> None:
    # Calculate embedding values for questions and answers
    for q in data["items"]:
        question_text = q["title"] + "\n" + q["body_markdown"]
        q["embedding"] = embeddings.embed_query(question_text)
        for a in q["answers"]:
            a["embedding"] = embeddings.embed_query(
                question_text + "\n" + a["body_markdown"]
            )

    import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
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
    input_text = st.text_input(
        "Which tag questions do you want to import?", value="neo4j"
    )
    return input_text


def get_pages():
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages (100 questions per page)", step=1, min_value=1
        )
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1)
    st.caption("Only questions with answers will be imported.")
    return (int(num_pages), int(start_page))


def render_page():
    st.header("StackOverflow Loader")
    st.subheader("Choose StackOverflow tags to load into Neo4j")
    st.caption("Go to http://localhost:7474/browser/ to explore the graph.")

    user_input = get_tag()
    num_pages, start_page = get_pages()

    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            try:
                for page in range(1, num_pages + 1):
                    load_so_data(user_input, start_page + (page - 1))
                st.success("Import successful", icon="✅")
            except Exception as e:
                st.error(f"Error: {e}", icon="🚨")
    with st.expander("Highly ranked questions rather than tags?"):
        if st.button("Import highly ranked questions"):
            with st.spinner("Loading... This might take a minute or two."):
                try:
                    load_high_score_so_data()
                    st.success("Import successful", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}", icon="🚨")


render_page()
