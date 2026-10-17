FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# State, logs and cached data live in /app/run so they survive container restarts.
WORKDIR /app/run
ENTRYPOINT ["python", "/app/main.py", "--config", "/app/config.json"]
CMD ["paper"]
