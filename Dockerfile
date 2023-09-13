FROM python:latest

COPY requirements.txt .
# COPY .env .
COPY app.py .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
