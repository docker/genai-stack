import os
from typing import List, Any

import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.graphs import Neo4jGraph

from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

if embedding_model_name == "ollama":
    embeddings = OllamaEmbeddings(base_url=ollama_base_url, model="llama2")
    dimension = 4096
    logger.info("Embedding: Using Ollama")
elif embedding_model_name == "openai":
    embeddings = OpenAIEmbeddings()
    dimension = 1536
    logger.info("Embedding: Using OpenAI")
else:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    dimension = 384
    logger.info("Embedding: Using SentenceTransformer")

create_vector_index(dimension)

if llm_name == "gpt-4":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    logger.info("LLM: Using GPT-4")
elif llm_name == "ollama":
    llm = ChatOllama(
        temperature=0, base_url=ollama_base_url, model="llama2", streaming=True
    )
    logger.info("LLM: Using Ollama (llama2)")
else:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    logger.info("LLM: Using GPT-3.5 Turbo")

# LLM only response
template = """
You are a helpful assistant that helps a support agent with answering programming questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def generate_llm_output(user_input: str, callbacks: List[Any]) -> str:
    answer = llm(
        chat_prompt.format_prompt(
            text=user_input,
        ).to_messages(),
        callbacks=callbacks,
    ).content
    return answer


# Rag response
neo4j_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="top_answers",  # vector by default
    text_node_property="body",  # text by default
    retrieval_query="""
    OPTIONAL MATCH (node)-[:ANSWERS]->(question)
    RETURN question.title + '\n' + question.body + '\n' + coalesce(node.body,"") AS text, score, {source:question.link} AS metadata
""",
)

general_system_template = """ 
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Each document in the context contains the source information.
For every document that you use information from, return their source link at the end of the answer.
----
{context}
----
"""
general_user_template = "Question:```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=neo4j_db.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)

# Rag + Knowledge Graph response
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
RETURN node.title + '\n' + node.body + '\n' + answerTexts AS text, score, {source:node.link} AS metadata
""",
)

kg_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=kg.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)

# Streamlit stuff
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 115px;
      background: white;
      z-index: 101;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


def chat_input():
    user_input = st.chat_input("What coding issue can I help you resolve today?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(name)
            stream_handler = StreamHandler(st.empty())
            result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )
            output = result  # ["answer"] + "\n" + result["sources"]
            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
            st.session_state[f"rag_mode"].append(name)


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:

        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []
    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        # Excluding the latest since it's streamed
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"Mode: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])


def mode_select() -> str:
    options = ["LLM only", "Vector", "Vector + Graph"]
    return st.radio("Select RAG mode", options, horizontal=True)


name = mode_select()
if name == "LLM only":
    output_function = generate_llm_output
elif name == "Vector":
    output_function = qa.run
elif name == "Vector + Graph":
    output_function = kg_qa.run

display_chat()
chat_input()
