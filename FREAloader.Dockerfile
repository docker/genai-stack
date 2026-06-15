FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY FREAloader.py .
COPY utils.py .
COPY chains.py .
COPY images ./images

EXPOSE 8506

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "FREAloader.py", "--server.port=8506", "--server.address=0.0.0.0"]
