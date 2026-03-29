# Official Playwright image with Chromium and dependencies pre-installed
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment settings for Render
ENV HEADLESS=true
ENV RENDER=true
ENV PYTHONUNBUFFERED=1

# Run the automation
CMD ["python", "main.py", "--headless"]
