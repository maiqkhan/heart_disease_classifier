FROM python:3.12.3-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY ./app/main.py ./models/log_regression.joblib ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "main:app", "--host" , "0.0.0.0", "--port", "9696"]