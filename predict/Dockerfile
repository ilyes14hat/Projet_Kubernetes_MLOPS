FROM python:3.9-slim
COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*



RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["flask", "run", "--host=127.0.0.1" , "--port=5000"]