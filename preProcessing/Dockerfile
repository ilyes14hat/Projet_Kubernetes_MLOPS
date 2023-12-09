FROM python:3.9-slim
COPY . /app
WORKDIR /app/
ADD . /app/
RUN  pip install -r requirements.txt
EXPOSE 5001
CMD ["python", "app.py"]

