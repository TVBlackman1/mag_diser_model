FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG POETRYVERSION='2.1.2'

RUN pip install poetry==${POETRYVERSION}

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

# ----------------------------------------------------------------------------------------
FROM python:3.11-slim as inference

RUN apt-get update && apt-get install -y \
    curl git build-essential libgcc1 libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG POETRYVERSION='2.1.2'

RUN pip install poetry==${POETRYVERSION}

COPY --from=builder /app/.venv /app/.venv

ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["poetry", "run", "python", "-m", "tests.test_enabled_features"]