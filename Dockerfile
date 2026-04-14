FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv pip install --system .

# /data is the persistent volume for metadata (index.db) and graph (graph.db).
# Mount a named volume or host directory here to survive container restarts.
VOLUME /data
WORKDIR /data

# Graph analysis is enabled by default; graph.db is stored alongside index.db
# in /data.  LLM enrichment is disabled by default — the container starts
# without any LLM API calls when ACI_LLM_* vars are absent.
ENV ACI_GRAPH_ENABLED=true \
    ACI_LLM_ENABLED=false \
    ACI_HTTP_ENABLED=false

ENTRYPOINT ["aci-mcp"]