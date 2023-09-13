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

# Read the wikipedia article
raw_documents = WikipediaLoader(query=page).load()

# Define chunking strategy
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=20
)
# Chunk the document
documents = text_splitter.split_documents(raw_documents)
# Remove the summary
for d in documents:
    del d.metadata["summary"]

neo4j_db = Neo4jVector.from_documents(
    documents,
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="wikipedia",  # vector by default
    node_label="WikipediaArticle",  # Chunk by default
    text_node_property="info",  # text by default
    embedding_node_property="vector",  # embedding by default
    create_id_index=True,  # True by default
)

neo4j_db.add_documents(
    [
        Document(
            page_content="LangChain is the coolest library since the Library of Alexandria",
            metadata={"author": "Tomaz", "confidence": 1.0}
        )
    ],
    ids=["langchain"],
)

result = neo4j_db.similarity_search(prompt, k=1)

print(result)