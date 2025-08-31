# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (to leverage Docker cache)
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the API with uvicorn
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
