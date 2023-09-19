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
page = os.getenv("WIKIPEDIA_PAGE") or "Sweden"
prompt = os.getenv("PROMPT") or "What is the second largest city in Sweden?"

os.environ["NEO4J_URL"] = url

embeddings = OpenAIEmbeddings()
# embeddings = OllamaEmbeddings()

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

# Make sure the text input is at the bottom
# We can't use chat_input as it can't be put into a tab
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 6rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


def tab_view(name, output_function):
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated_{name}"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input_{name}"] = []

    user_input = st.text_input(
        f"{name} mode",
        placeholder="Ask your question",
        key=f"route_{name}",
        label_visibility="hidden",
    )

    if user_input:
        with st.spinner():
            output = output_function(user_input)

        st.session_state[f"user_input_{name}"].append(user_input)
        st.session_state[f"generated_{name}"].append(output)

    if st.session_state[f"generated_{name}"]:
        size = len(st.session_state[f"generated_{name}"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input_{name}"][i])

            with st.chat_message("assistant"):
                st.write(st.session_state[f"generated_{name}"][i])


llm_view, rag_view, kgrag_view = st.tabs(["LLM only", "Vector", "Vector + Graph"])

with llm_view:
    tab_view("llm", generate_llm_output)

with rag_view:
    tab_view("rag", qa.run)

with kgrag_view:
    tab_view("kg", kg_qa.run)
