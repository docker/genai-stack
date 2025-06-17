import streamlit as st
import logging
from streamlit.logger import get_logger
from rag_utils.config import init
from rag_utils.pipeline import RAG_document_retrieval
from utils import StreamHandler
import base64


logging.basicConfig(level = logging.INFO)

logger = get_logger(__name__)

if 'init' not in st.session_state:
    st.session_state.init = True
    init()
    
st.set_page_config(page_title="StartLegal - Anexar a Minuta", layout="wide")



st.subheader(
    "Anexe a minuta da escritura para iniciar a revisão.",
    divider='gray'
)

# upload a your files
uploaded_file_minuta = st.file_uploader(
    "Suba o documento da Minuta em formato PDF.", 
    accept_multiple_files=False,
    type="pdf"
)

if uploaded_file_minuta:
    st.write("A IA irá coletar as informações presentes no documento...")
    
    col1, col2 = st.columns(2, vertical_alignment="center")

    with col2:
        base64_pdf = base64.b64encode(uploaded_file_minuta.getvalue()).decode("utf-8")
        pdf_display = (
            f'<embed src="data:application/pdf;base64,{base64_pdf}" '
            'width="960" height="2160" type="application/pdf"></embed>'
        )
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.session_state.minuta_file = uploaded_file_minuta

    with col1:
        # Collect and structure data from Buyers 
        answer = RAG_document_retrieval(
                    document='Minuta Comprador',
                    file=uploaded_file_minuta,
                    prompts=st.session_state.prompts,
                    logger=logger,
                    embeddings=st.session_state.embeddings,
                    vectordb_config=st.session_state.vectorstore_config,
                    llm=st.session_state.llm,
                    ocr_params={
                        'pages': [0],
                        'lang': 'por'
                    }
                )
        
        st.session_state.minuta_comprador = answer

        # Print output answer
        stream_handler = StreamHandler(st.empty())
        for token in st.session_state.minuta_comprador:
            stream_handler.on_llm_new_token(token=token)

        # Collect and structure data from Sellers 
        answer = RAG_document_retrieval(
                    document='Minuta Vendedor',
                    file=uploaded_file_minuta,
                    prompts=st.session_state.prompts,
                    logger=logger,
                    embeddings=st.session_state.embeddings,
                    vectordb_config=st.session_state.vectorstore_config,
                    llm=st.session_state.llm,
                    ocr_params={
                        'pages': [0],
                        'lang': 'por'
                    }
                )

        st.session_state.minuta_vendedor = answer

        # Print output answer
        stream_handler2 = StreamHandler(st.empty())
        for token in st.session_state.minuta_vendedor:
            stream_handler2.on_llm_new_token(token=token)

        # Collect and structure data from Real State/Land
        answer = RAG_document_retrieval(
                    document='Minuta Imóvel',
                    file=uploaded_file_minuta,
                    prompts=st.session_state.prompts,
                    logger=logger,
                    embeddings=st.session_state.embeddings,
                    vectordb_config=st.session_state.vectorstore_config,
                    llm=st.session_state.llm,
                    ocr_params={
                        'pages': [0,1],
                        'lang': 'por'
                    }
                )

        st.session_state.minuta_imovel = answer

        # Print output answer
        stream_handler3 = StreamHandler(st.empty())
        for token in st.session_state.minuta_imovel:
            stream_handler3.on_llm_new_token(token=token)

else:
    if 'minuta_file' in st.session_state:
        col3, col4 = st.columns(2, vertical_alignment="center")

        with col4:
            base64_pdf = base64.b64encode(st.session_state.minuta_file.getvalue()).decode("utf-8")
            pdf_display = (
                f'<embed src="data:application/pdf;base64,{base64_pdf}" '
                'width="960" height="2160" type="application/pdf"></embed>'
            )
            
            st.markdown(pdf_display, unsafe_allow_html=True)
        
        with col3:
            if 'minuta_comprador' in st.session_state:
                st.write(st.session_state.minuta_comprador)

            if 'minuta_vendedor' in st.session_state:
                st.write(st.session_state.minuta_vendedor)

            if 'minuta_imovel' in st.session_state:
                st.write(st.session_state.minuta_imovel)