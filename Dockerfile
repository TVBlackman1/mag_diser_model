FROM python:3.12-slim


WORKDIR /app

# Установим зависимости системы
RUN apt-get update && apt-get install -y \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Установим poetry
ENV POETRY_VERSION=1.8.2

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Установим рабочую директорию
WORKDIR /app

# Скопируем только файлы зависимостей для кэширования
COPY pyproject.toml poetry.lock* ./

# Установим зависимости проекта через poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

RUN pip install --no-cache-dir tensorflow keras numpy matplotlib opencv-python pygame

CMD ["poetry", "run python -m inference.test_loop"]