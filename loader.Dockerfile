FROM python:latest

COPY requirements.txt .
# COPY .env .
COPY loader.py .

RUN pip install -r requirements.txt

CMD ["python", "loader.py"]
