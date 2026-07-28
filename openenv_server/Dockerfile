FROM python:3.11-slim

WORKDIR /app

COPY openenv_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY raaga_env/ ./raaga_env/
COPY openenv_server/server.py ./openenv_server/server.py

EXPOSE 7860

CMD ["python", "openenv_server/server.py"]
