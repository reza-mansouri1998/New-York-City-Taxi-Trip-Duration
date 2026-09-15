# Use the exact Python version to prevent Pickle deserialization errors
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Ensure Python recognizes the 'src' directory as a package
ENV PYTHONPATH=/app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the production requirements file into the container
COPY requirements.txt .

# Install packages without saving the cache to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 for the web server
EXPOSE 5000

# Run Gunicorn for production-grade serving
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]