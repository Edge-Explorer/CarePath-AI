# --------------------------
# Stage 1: Base image
# --------------------------
FROM python:3.10-slim

# Prevent Python from buffering stdout and stdin (useful in logs)
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system deps (for FAISS, sqlite, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (we’ll create requirements.txt below)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
