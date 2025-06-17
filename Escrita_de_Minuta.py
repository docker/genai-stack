import streamlit as st

st.title(body='✍️ StartLegal - Escritor de Minutas')
st.header("Assistente de Elaboração de Escrituras", divider='gray', )

st.write(
    "Anexe os documentos necessários das partes compradora e vendedora e a escritura do imóvel."
)

parte_compradora = st.container()

parte_compradora.markdown("**Parte Compradora**")
parte_compradora.file_uploader(
    "Anexe os documentos da parte compradora",
    type=["pdf", "jpg", "jpeg", "png"],
    key="parte_compradora",
    accept_multiple_files=True
)

st.divider()

parte_vendedora = st.container()

parte_vendedora.markdown("**Parte Vendedora**")
parte_vendedora.file_uploader(
    "Anexe os documentos da parte vendedora",
    type=["pdf", "jpg", "jpeg", "png"],
    key="parte_vendedora",
    accept_multiple_files=True
)

st.divider()

imovel = st.container()

imovel.markdown("**Escritura do Imóvel**")
imovel.file_uploader(
    "Anexe a escritura do imóvel",
    type=["pdf", "jpg", "jpeg", "png"], 
    key="imovel",
    accept_multiple_files=True
)