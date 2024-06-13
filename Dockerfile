FROM python:3.11.4

WORKDIR /project

COPY . /project

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
