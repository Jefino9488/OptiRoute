FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY data/ ./data/
COPY main.py ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Expose port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
