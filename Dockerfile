# Base image

FROM python:3.10-slim
 
# Set working directory

WORKDIR /app
 
# Copy required files

COPY predict.py .

COPY autompg_model.bin .

COPY requirements.txt .
 
 
# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Expose port

EXPOSE 8000
 
# Command to run the app

CMD ["python", "predict.py"]

 