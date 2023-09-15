FROM python:latest

COPY requirements.txt .

RUN pip install -r requirements.txt

# COPY .env .
COPY loader.py .

CMD ["python", "loader.py"]
