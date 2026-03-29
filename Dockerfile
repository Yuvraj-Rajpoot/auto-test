# Use official Playwright Python image that matches a supported version
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /app

# Copy and install Python dependencies (including your pinned playwright==1.40.0)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Environment variables for Render
ENV HEADLESS=true
ENV RENDER=true
ENV PYTHONUNBUFFERED=1

# Run your automation script
CMD ["python", "main.py", "--headless"]
