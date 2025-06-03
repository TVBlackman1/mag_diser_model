FROM python:3.12-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y \
    curl build-essential git \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установим poetry
ENV POETRY_VERSION=2.1.2
ENV PATH="/root/.local/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Установим рабочую директорию
WORKDIR /app

# Копируем только файлы зависимостей
COPY pyproject.toml poetry.lock* ./

# Установка зависимостей без виртуального окружения
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Копируем оставшийся проект
COPY . .

# Запуск через poetry
CMD ["poetry", "run", "python", "-m", "inference.test_loop"]