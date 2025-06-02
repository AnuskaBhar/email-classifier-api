FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]