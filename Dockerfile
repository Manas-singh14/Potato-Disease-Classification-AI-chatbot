# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first
# (Docker caches this layer — speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything — code, model, chat folder
COPY . .

# Expose port
EXPOSE 10000

# Run the app


CMD ["uvicorn", "backend_final_groq:app", "--host", "0.0.0.0", "--port", "10000"]