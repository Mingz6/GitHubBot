FROM python:3.11-slim

WORKDIR /app

# Install only required system dependencies (remove software-properties-common which isn't in slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Copy requirements and install Python dependencies (upgrade pip first)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default API key (this will be overridden by Hugging Face secrets)
ENV TOGETHER_API_KEY=""

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]