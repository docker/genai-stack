import streamlit as st
from streamlit.logger import get_logger
import logging
from utils import StreamHandler
from rag_utils.config import init
from rag_utils.pipeline import RAG_document_retrieval, RAG_document_validator
import base64


logging.basicConfig(level = logging.INFO)

logger = get_logger(__name__)

if 'init' not in st.session_state:
    st.session_state.init = True
    init()

st.session_state.owner_documents_list = [
    'CNH Vendedor', 
    'Comprovante de Residência Vendedor',
    'Matrícula do Imóvel'
]

if 'init_owner_review_page' not in st.session_state:
    st.session_state.init_owner_review_page = True
    st.session_state.final_answer_owner = dict().fromkeys(st.session_state.owner_documents_list)

# Define a list of Documents at app init() method
tabs = st.tabs(st.session_state.owner_documents_list)

for tab, document in zip(tabs, st.session_state.owner_documents_list):
    with tab:
        # upload a your files
        uploaded_file = st.file_uploader(
            "Suba o documento em algum desses formatos: PDF, png, jpeg, ou txt.", 
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            key=document
        )
    
        if uploaded_file:
            st.write("A IA irá coletar e validar as informações presentes...")

            col1, col2, col3 = st.columns(3, vertical_alignment="top")

            with col1:
                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                pdf_display = (
                    f'<embed src="data:application/pdf;base64,{base64_pdf}" '
                    'width="640" height="1080" type="application/pdf"></embed>'
                )
                
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            with col3:
                base64_pdf = base64.b64encode(st.session_state.minuta_file.getvalue()).decode("utf-8")
                pdf_display = (
                    f'<embed src="data:application/pdf;base64,{base64_pdf}" '
                    'width="640" height="1080" type="application/pdf"></embed>'
                )
                
                st.markdown(pdf_display, unsafe_allow_html=True)

            with col2:
                answer = RAG_document_retrieval(
                            document=document,
                            file=uploaded_file,
                            prompts=st.session_state.prompts,
                            logger=logger,
                            embeddings=st.session_state.embeddings,
                            vectordb_config=st.session_state.vectorstore_config,
                            llm=st.session_state.llm
                        )
                # Print output answer
                stream_handler = StreamHandler(st.empty())
                for token in answer:
                    stream_handler.on_llm_new_token(token=token)

                # Ask to LLM a table showing the Document data and Minuta data
                st.write(f"Validando dados de {document} com os dados da Minuta.")

                minuta_answer = st.session_state.minuta_vendedor 
                if document == 'Matrícula do Imóvel':
                    minuta_answer = st.session_state.minuta_imovel
                
                final_answer = RAG_document_validator(
                    document=document,
                    document_answer=answer,
                    minuta_answer=minuta_answer,
                    llm=st.session_state.llm,
                    logger=logger
                )
                st.session_state.final_answer_owner[document] = final_answer
                
                # Print output answer
                stream_handler = StreamHandler(st.empty())
                for token in final_answer:
                    stream_handler.on_llm_new_token(token=token)
            
        else:
            if st.session_state.final_answer_owner[document]:
                st.write(st.session_state.final_answer_owner[document])