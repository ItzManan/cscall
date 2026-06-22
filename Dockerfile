FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[live]"

EXPOSE 8000

CMD ["python", "-m", "cscall.cli", "live", "--host", "0.0.0.0", "--port", "8000"]
