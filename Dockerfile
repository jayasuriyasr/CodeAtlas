# The SEI server, for running next to docker compose's Neo4j.
#   docker compose --profile app up -d     (see the `sei` service in docker-compose.yml)
FROM python:3.12-slim

# git: the indexer asks it for the file list, so .gitignore is respected.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# The project is mounted read-only and owned by someone else; let git read it.
RUN git config --system --add safe.directory '*'

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 SEI_CACHE_PATH=/data/embeddings.sqlite
EXPOSE 8000
# Inside a container the server must bind 0.0.0.0 to be reachable, which the
# CLI allows only with an SEI_API_TOKEN set (the compose service requires one).
ENTRYPOINT ["sei"]
CMD ["serve", "/project", "--host", "0.0.0.0", "--port", "8000"]
