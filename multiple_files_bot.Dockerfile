FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-por

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

ADD rag_utils rag_utils
ADD pages pages
COPY __init__.py .
COPY StartLegal.py .
COPY Revisor_de_Minuta.py .
COPY Escrita_de_Minuta.py .
COPY prompts.json .
COPY utils.py .
COPY chains.py .

EXPOSE 8505

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "StartLegal.py", "--server.port=8505", "--server.address=0.0.0.0"]
