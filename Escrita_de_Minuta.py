import streamlit as st
from rag_utils.config import init
from rag_utils.pipeline import RAG_document_retrieval
import base64
import threading
import logging
import time

session_state_status_percent = 0


def parte_compradora_agents_thread(uploaded_files):
    if uploaded_files:
        st.session_state.status = "Processando documentos da parte compradora..."
        global session_state_status_percent
        session_state_status_percent = 0
        len_uploaded_files = len(uploaded_files)
        logging.info("Parte compradora: Iniciando o processamento dos documentos.")
        
        # Simulate processing each uploaded file
        for p, uploaded_file in enumerate(uploaded_files):
            # Simulate processing time
            time.sleep(1)
            st.session_state.status = f"Processando {uploaded_file.name}..."
            session_state_status_percent = (session_state_status_percent+p+1) / len_uploaded_files
            logging.info(f"Parte compradora: Processando {uploaded_file.name}...")
            logging.info(f"Parte compradora (Thread): Progresso {session_state_status_percent:.2%}")

            # Here you would typically call your RAG_document_retrieval function
            # For example: RAG_document_retrieval(uploaded_file)
        
        st.session_state.status = "Documentos da parte compradora processados com sucesso!"
        logging.info("Parte compradora: Documentos processados com sucesso!")


def parte_compradora_button_callback(uploaded_files, container):
    global session_state_status_percent
    
    thread = threading.Thread(
        target=parte_compradora_agents_thread,
        args=(uploaded_files,),
        daemon=True
    )
    thread.start()
    
    with container:
        bar = st.progress(0, text_ocr)
        while session_state_status_percent*100 < 100:
            time.sleep(0.1)
            bar.progress(session_state_status_percent, text_ocr)
            logging.info(f"Parte compradora: Progresso {session_state_status_percent:.2%}")
        bar.empty()
        thread.join()
    st.session_state.status = "Processamento finalizado!"
    logging.info("Parte compradora: Processamento finalizado!")


def parte_vendedora_button_callback():
    pass


def imovel_button_callback():
    pass


def container_files_uploader_and_text_writer(container, labels: dict, key, callback):
    container.markdown(f"**{labels['markdown_label']}**")
    
    uploaded_files = container.file_uploader(
        labels['file_uploader_label'],
        type=["pdf", "jpg", "jpeg", "png"],
        key=f"{key}_file_uploader",
        accept_multiple_files=True
    )
    
    write_text_button = container.button(
        labels['button_label'],
        help="Clique para gerar o parágrafo com as informações extraídas dos documentos.",
        disabled=not uploaded_files,
        on_click=callback,
        args=(uploaded_files, container),
        key=f"{key}_button"
    )
    
    if uploaded_files and write_text_button:
        container.write(f"Status: {st.session_state.status}")

logging.basicConfig(level = logging.INFO)

if 'init' not in st.session_state:
    st.session_state.init = True
    if 'status' not in st.session_state:
        st.session_state.status = "Aguardando o upload dos documentos..."
    init()

if 'init_writer_page' not in st.session_state:
    st.session_state.init_buyer_writer_page = True

    st.session_state.buyer_documents_list = [
        'CNH Comprador', 
        'Comprovante de Residência Comprador', 
        'Certidão de Casamento Comprador',
        'Pacto Antenupcial ou Declaração de União Estável',
        'CNH Cônjuge',
        'Quitação ITBI'
    ]
    
    st.session_state.owner_documents_list = [
        'CNH Vendedor',
        'Comprovante de Residência Vendedor',
        'Matrícula do Imóvel'
    ]

text_ocr = "Extraindo informações dos documentos..."

st.title(body='✍️ StartLegal - Escritor de Minutas')
st.header("Assistente de Elaboração de Escrituras", divider='gray', )

st.write(
    "Anexe os documentos necessários das partes compradora e vendedora e a escritura do imóvel."
)

parte_compradora = st.container()

container_files_uploader_and_text_writer(
    container=parte_compradora,
    labels={
        'markdown_label': '**Parte Compradora**',
        'file_uploader_label': 'Anexe os documentos da parte compradora',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='parte_compradora',
    callback=parte_compradora_button_callback
)

st.divider()

parte_vendedora = st.container()

container_files_uploader_and_text_writer(
    container=parte_vendedora,
    labels={
        'markdown_label': '**Parte Vendedora**',
        'file_uploader_label': 'Anexe os documentos da parte vendedora',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='parte_vendedora',
    callback=parte_vendedora_button_callback
)

st.divider()

imovel = st.container()

container_files_uploader_and_text_writer(
    container=imovel,
    labels={
        'markdown_label': '**Escritura do Imóvel**',
        'file_uploader_label': 'Anexe a escritura do imóvel',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='imovel',
    callback=imovel_button_callback
)
