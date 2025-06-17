import streamlit as st
from streamlit.logger import get_logger


st.set_page_config(page_title="StartLegal - IA para Cartórios", page_icon="🤖", layout="wide")

logger = get_logger(__name__)

escritor_page = st.Page("Escrita_de_Minuta.py", title="Escrita de Minuta", icon="✍️")

revisor_page = st.Page("Revisor_de_Minuta.py", title="Guia de Usabilidade", icon="📄")
upload_minuta_page = st.Page("pages/1_Anexar_Minuta.py", title="Minuta", icon="📄")
parte_compradora_page = st.Page("pages/2_Parte_Compradora.py", title="Parte Compradora", icon="📄")
parte_vendedora_page = st.Page("pages/3_Parte_Vendedora.py", title="Parte Vendedora", icon="📄")

pg = st.navigation(
    {
        "Escrita de Minutas": [escritor_page],
        "Revisão de Minutas": [revisor_page, upload_minuta_page, parte_compradora_page, parte_vendedora_page],
    }
)
pg.run()