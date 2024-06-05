import os
import requests
import mimetypes
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model
from utils import create_constraints, create_vector_index
from PIL import Image


load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

#results = read_files_info('C:/SyncedFolder/Team Shares/FREA/')

#so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"
#next(results)
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)

def read_files_info(directory='.'):
    #files_info = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            info = os.stat(file_path)
            file_info = {
                'path': file_path,
                'name': filename,
                'type': mimetypes.guess_type(file_path)[0],
                'size': os.path.getsize(file_path),
                'creation_time': info.st_ctime, 
                'modification_time': info.st_mtime  
            }
            yield file_info

def get_file_info(file_path):
    file_info = next(results)
    switch_case(file_info['type'])  


def switch_case(value):
    switch = {
        'text/html': funcWebPages,
        'text/markdown': funcMarkdown,
        'application/xml': funcXML,
        'application/pdf': funcPDF,
        'application/msword': funcDOC,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': funcDOCX,
        'application/vnd.ms-excel (XLS)': funcXLS,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': funcXLSX,
        'application/vnd.ms-powerpoint (PPT)': funcPPT,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': funcPPTX,
        'application/rtf': funcRTF,
        'image/jpeg': funcJPG,
        'image/png': funcPNG,
        'image/gif': funcGIF,
        'image/bmp': funcBMP,
        'image/tiff': funcTIFF,
        'application/javascript': funcJavaScript,
        'application/zip': funcZIP,
        'application/gzip': funcGZIP,
        'audio/mpeg': funcMP3,
        'video/mp4': funcMP4,
        'audio/wav': funcWAV,
        'audio/ogg': funcOGG,
        'video/webm': funcWEBM,
        'application/json': funcJSON,
        'application/x-yaml': funcYAML,
        'application/epub+zip': funcEPUB,
        'application/x-mobipocket-ebook': funcMOBI
    }
    func = switch.get(value, default_function)
    func()

def load_so_data(tag: str = "neo4j", page: int = 1) -> None:
     parameters = (
   
     )
    #data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)


def load_high_score_so_data() -> None:
    parameters = (
        
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

    # Cypher, the query language of Neo4j, is used to import the data
    # https://neo4j.com/docs/getting-started/cypher-intro/
    # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
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
    datamodel_image = Image.open("./images/datamodel.png")
    st.header("StackOverflow Loader")
    st.subheader("Choose StackOverflow tags to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")

    user_input = get_tag()
    num_pages, start_page = get_pages()

    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            try:
                for page in range(1, num_pages + 1):
                    load_so_data(user_input, start_page + (page - 1))
                st.success("Import successful", icon="✅")
                st.caption("Data model")
                st.image(datamodel_image)
                st.caption("Go to http://localhost:7474/ to interact with the database")
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
