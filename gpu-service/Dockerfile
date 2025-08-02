FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
COPY requirements_gpu.txt .
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# Copy app code
COPY . .

EXPOSE 8080
CMD ["uvicorn", "main_gpu:app", "--host", "0.0.0.0", "--port", "8080"]
