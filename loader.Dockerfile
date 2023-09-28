# TODO Langchain base image
FROM python:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

# COPY .env .
COPY loader.py .
COPY utils.py .

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "loader.py", "--server.port=8502", "--server.address=0.0.0.0"]
