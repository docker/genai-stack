import os
from typing import List, Any

import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import extract_title_and_question

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


def create_vector_index(dimension: int) -> None:
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
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
    )
    dimension = 384
    logger.info("Embedding: Using SentenceTransformer")

create_vector_index(dimension)

if llm_name == "gpt-4":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    logger.info("LLM: Using GPT-4")
elif llm_name == "ollama":
    llm = ChatOllama(
        temperature=0,
        base_url=ollama_base_url,
        model="llama2",
        streaming=True,
        top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=3072,  # Sets the size of the context window used to generate the next token.
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


def generate_llm_output(
    user_input: str, callbacks: List[Any], prompt=chat_prompt
) -> str:
    answer = llm(
        prompt.format_prompt(
            text=user_input,
        ).to_messages(),
        callbacks=callbacks,
    ).content
    return {"answer": answer}


# Vector response
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
    RETURN 'Question: ' + question.title + '\n' + question.body + '\nAnswer: ' + 
            coalesce(node.body,"") AS text, score, {source:question.link} AS metadata
    ORDER BY score ASC // so that best answer are the last
""",
)

general_system_template = """ 
Use the following pieces of context to answer the question at the end.
The context contains question-answer pairs and their links from Stackoverflow.
You should prefer information from accepted or more upvoted answers.
Make sure to rely on information from the answers and not on questions to provide accuate responses.
When you find particular answer in the context useful, make sure to cite it in the answer using the link.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----
{summaries}
----
Each answer you generate should contain a section at the end of links to 
Stackoverflow questions and answers you found useful, which are described under Source value.
You can only use links to StackOverflow questions that are present in the context and always
add links to the end of the answer in the style of citations.
Generate concise answers with references sources section of links to 
relevant StackOverflow questions only at the end of the answer.
"""
general_user_template = "Question:```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

qa_chain = load_qa_with_sources_chain(
    llm,
    chain_type="stuff",
    prompt=qa_prompt,
)
qa = RetrievalQAWithSourcesChain(
    combine_documents_chain=qa_chain,
    retriever=neo4j_db.as_retriever(search_kwargs={"k": 2}),
    reduce_k_below_max_tokens=True,
    max_tokens_limit=3375,
)

# Vector + Knowledge Graph response
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
    WITH collect(a)[..2] as answers
    RETURN reduce(str='', a IN answers | str + 
            '\n### Answer (Accepted: '+ a.is_accepted +' Score: ' + a.score+ '): '+  a.body + '\n') as answerTexts
} 
RETURN '##Question: ' + node.title + '\n' + node.body + '\n' 
       + answerTexts AS text, score, {source: node.link} AS metadata
ORDER BY score ASC // so that best answers are the last
""",
)

kg_qa = RetrievalQAWithSourcesChain(
    combine_documents_chain=qa_chain,
    retriever=kg.as_retriever(search_kwargs={"k": 2}),
    reduce_k_below_max_tokens=True,
    max_tokens_limit=3375,
)

# Streamlit UI
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 115px;
      background: white;
      z-index: 101;
    }}

    /* Generate ticket text area */
    textarea[aria-label="Description"] {{
        height: 200px;
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
            st.caption(f"RAG: {name}")
            stream_handler = StreamHandler(st.empty())
            result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )["answer"]
            output = result
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
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.expander("Not finding what you're looking for?"):
            st.write(
                "Automatically generate a draft for an internal ticket to our support team."
            )
            st.button(
                "Generate ticket",
                type="primary",
                key="show_ticket",
                on_click=open_sidebar,
            )
        with st.container():
            st.write("&nbsp;")


def mode_select() -> str:
    options = ["Disabled", "Enabled"]
    return st.radio("Select RAG mode", options, horizontal=True)


name = mode_select()
if name == "LLM only" or name == "Disabled":
    output_function = generate_llm_output
elif name == "Vector":
    output_function = qa
elif name == "Vector + Graph" or name == "Enabled":
    output_function = kg_qa


def generate_ticket():
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. {question[0]}\n"
        questions_prompt += f"{question[1]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Can youformulate a question in the same style, detail and tone as the following example questions?
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return example:
    ---
    Title: How do I use the Neo4j Python driver?
    Question: I'm trying to connect to Neo4j using the Python driver, but I'm getting an error.
    ---
    
    Here's the question to rewrite:
    
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    q_prompt = st.session_state[f"user_input"][-1]
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt, HumanMessagePromptTemplate.from_template("{text}")]
    )
    llm_response = generate_llm_output(q_prompt, [], chat_prompt)
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)


def open_sidebar():
    st.session_state.open_sidebar = True


def close_sidebar():
    st.session_state.open_sidebar = False


if not "open_sidebar" in st.session_state:
    st.session_state.open_sidebar = False
if st.session_state.open_sidebar:
    new_title, new_question = generate_ticket()
    with st.sidebar:
        st.title("Ticket draft")
        st.write("Auto generated draft ticket")
        st.text_input("Title", new_title)
        st.text_area("Description", new_question)
        st.button(
            "Submit to support team",
            type="primary",
            key="submit_ticket",
            on_click=close_sidebar,
        )


display_chat()
chat_input()
