import os

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv('.env')

url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')
page = os.getenv('WIKIPEDIA_PAGE') or "Sweden"
prompt = os.getenv('PROMPT') or "What is the second largest city in Sweden?"

os.environ["NEO4J_URL"] = url

embeddings = OpenAIEmbeddings()

neo4j_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="wikipedia",  # vector by default
    node_label="WikipediaArticle",  # Chunk by default
    text_node_property="info",  # text by default
    embedding_node_property="vector",  # embedding by default
    create_id_index=False,  # True by default
    # todo retrieval query for KG
)

result = neo4j_db.similarity_search(prompt, k=1)

print(result)