FROM mcr.microsoft.com/playwright/python:v1.56.0-noble

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.8.5
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY webvoyeur ./webvoyeur
COPY README.md LICENSE ./

RUN poetry install --only main --no-interaction --no-ansi \
    && python -m playwright install chromium firefox

RUN mkdir -p /app/output

ENTRYPOINT ["webvoyeur"]
CMD ["--help"]