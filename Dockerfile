FROM python:3.11

ENV PORT 8080

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
