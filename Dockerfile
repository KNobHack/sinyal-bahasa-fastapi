FROM python:3.10.3-slim-buster

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

CMD ["python", "main.py"]
