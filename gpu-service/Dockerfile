FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and required system libraries for Matplotlib
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libfreetype6 libxft2 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_gpu.txt .
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# Copy app code
COPY . .

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main_gpu:app --host 0.0.0.0 --port ${PORT:-8080}"]


