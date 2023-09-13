# TODO Langchain base image
FROM python:latest

COPY requirements.txt .
# COPY .env .
COPY bot.py .

RUN pip install -r requirements.txt

CMD ["python", "bot.py"]
