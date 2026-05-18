# ── Dockerfile for Hugging Face Spaces / any container platform ──────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860
ENV PORT=7860
ENV FLASK_DEBUG=false

# Start with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
