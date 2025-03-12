# Use a lightweight PyTorch base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/models" \
    HF_HOME="/models"

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
