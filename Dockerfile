# Use official Playwright Python image (includes Chromium + all deps)
FROM mcr.microsoft.com/playwright/python:v1.40.0-noble

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Force headless mode on Render
ENV HEADLESS=true
ENV RENDER=true
ENV PYTHONUNBUFFERED=1

# Run the automation script
CMD ["python", "main.py", "--headless"]
