FROM python:3.10-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (TensorFlow sometimes needs these)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the app source code
COPY . .

# Expose port (Render looks for this)
EXPOSE 5000

# Run app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
