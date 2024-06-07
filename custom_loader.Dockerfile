FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY loader.py .
COPY custom_loader.py .
COPY utils.py .
COPY chains.py .
COPY images ./images

ENTRYPOINT ["python", "custom_loader.py"]
