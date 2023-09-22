import os

import streamlit as st
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

os.environ["NEO4J_URL"] = url

if embedding_model_name == "ollama":
    embeddings = OllamaEmbeddings(base_url=ollama_base_url, model="llama2")
    print("Embedding: Using Ollama")
elif embedding_model_name == "openai":
    embeddings = OpenAIEmbeddings()
    print("Embedding: Using OpenAI")
else:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding: Using SentenceTransformer")
    
if llm_name == "gpt-4":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    print("LLM: Using GPT-4")
elif llm_name == "ollama":
    llm = ChatOllama(temperature=0, base_url=ollama_base_url, model="llama2")
    print("LLM: Using Ollama (llama2)")
else:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    print("LLM: Using GPT-3.5 Turbo")

# LLM only response
template = "You are a helpful assistant that helps with programming questions."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def generate_llm_output(user_input: str) -> str:
    answer = llm(
        chat_prompt.format_prompt(
            text=user_input,
        ).to_messages()
    ).content
    return {"answer":answer,"sources":""}


# Rag response
neo4j_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="stackoverflow",  # vector by default
    text_node_property="body",  # text by default
    retrieval_query="""
    OPTIONAL MATCH (node)<-[:ANSWERS]-(a)
    RETURN node.title + '\n' + node.body + '\n' + coalesce(a.text,"") AS text, score, {source:node.link} AS metadata LIMIT 1
""",
)

qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff", retriever=neo4j_db.as_retriever(search_kwargs={"k": 2})
)

# Rag + KG
kg = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="stackoverflow",  # vector by default
    text_node_property="body",  # text by default
    retrieval_query="""
CALL  { with node
    MATCH (node)<-[:ANSWERS]-(a)
    WITH a
    ORDER BY a.is_accepted DESC, a.score DESC
    WITH collect(a.body)[..2] as answers
    RETURN reduce(str='', text IN answers | str +  text + '\n') as answerTexts
} 
RETURN node.body + '\n' + answerTexts AS text, score, {source:node.link} AS metadata
""",
)

kg_qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff", retriever=kg.as_retriever(search_kwargs={"k": 2})
)

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
        try:
            result = output_function(user_input)
            if not type(result) is list: result=[result]
            output = "\n".join([data["answer"] + "\n" + data["sources"] for data in result])
        except (KeyError, TypeError): # todo should we really have control-flow by exception?
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
if name == "LLM only":
    output_function = generate_llm_output
elif name == "Vector":
    output_function = qa
elif name == "Vector + Graph":
    output_function = kg_qa

chat_input()
display_chat()
