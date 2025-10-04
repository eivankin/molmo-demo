FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

ENV PYTHONWARNINGS=ignore

EXPOSE 7860

CMD ["python", "app.py"]