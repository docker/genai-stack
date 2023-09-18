import os

import streamlit as st
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
page = os.getenv("WIKIPEDIA_PAGE") or "Sweden"
prompt = os.getenv("PROMPT") or "What is the second largest city in Sweden?"

os.environ["NEO4J_URL"] = url

embeddings = OpenAIEmbeddings()
# embeddings = OllamaEmbeddings()

neo4j_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="stackoverflow",  # vector by default
    node_label="Question",  # Chunk by default
    text_node_property="body",  # text by default
    embedding_node_property="embedding",  # embedding by default
    # todo retrieval query for KG
)

# result = neo4j_db.similarity_search(prompt, k=1)
#
# print(result)
#
# res = embeddings.embed_query(prompt)
# print(len(res))


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0), neo4j_db.as_retriever(), memory=memory
)


# Streamlit stuff

# Session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = []


def get_text() -> str:
    input_text = st.chat_input("Ask away?")
    return input_text


user_input = get_text()

if user_input:
    output = qa.run(user_input)

    st.session_state.user_input.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    size = len(st.session_state["generated"])
    # Display only the last three exchanges
    for i in range(max(size - 3, 0), size):
        with st.chat_message("user"):
            st.write(st.session_state["user_input"][i])

        with st.chat_message("assistant"):
            st.write(st.session_state["generated"][i])
