FROM python:3.10-slim

# Install system libraries needed by OpenCV and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, models, templates, and static assets
COPY . .

# Set default port (Railway injects PORT environment variable)
ENV PORT=8080
EXPOSE 8080

# Run FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
