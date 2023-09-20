import os

import streamlit as st
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

os.environ["NEO4J_URL"] = url

# embeddings = OllamaEmbeddings(base_url=ollama_base_url)
embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0)

# LLM only response
template = "You are a helpful assistant that helps with programming questions."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def generate_llm_output(user_input: str) -> str:
    return llm(
        chat_prompt.format_prompt(
            text=user_input,
        ).to_messages()
    ).content


# Rag response
neo4j_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="stackoverflow",  # vector by default
    text_node_property="body",  # text by default
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, neo4j_db.as_retriever(), memory=memory)

# Rag + KG
kg = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="stackoverflow",  # vector by default
    text_node_property="body",  # text by default
    retrieval_query="RETURN 'fancy' AS text, 1 AS score, {} AS metadata",  # Fix this
)

kg_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
kg_qa = ConversationalRetrievalChain.from_llm(llm, kg.as_retriever(), memory=kg_memory)

# Streamlit stuff
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select sophistication mode"]) {{
      position: fixed;
      bottom: 115px;
      background: white;
      z-index: 101;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

def chat_input():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    user_input = st.chat_input("What coding issue can I help you resolve today?")

    if user_input:
        output = output_function(user_input)
        st.session_state[f"user_input"].append(user_input)
        st.session_state[f"generated"].append(output)
        st.session_state[f"rag_mode"].append(name)

def display_chat():
    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"Mode: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

def mode_select() -> str:
    options = ["LLM only", "Vector", "Vector + Graph"]
    return st.radio("Select sophistication mode", options, horizontal=True)

name = mode_select()
if(name == "LLM only"):
    output_function = generate_llm_output
elif(name == "Vector"):
    output_function = qa.run
elif(name == "Vector + Graph"):
    output_function = kg_qa.run

chat_input()
display_chat()


