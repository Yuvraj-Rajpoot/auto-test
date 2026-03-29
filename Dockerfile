FROM python:3.11-slim

# Install all system dependencies required by Playwright + Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libxcomposite1 \
    libxdamage1 \
    libatk1.0-0 \
    libasound2 \
    libdbus-1-3 \
    libnspr4 \
    libgbm1 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libnss3 \
    libxrandr2 \
    libasound2-plugins \
    fonts-liberation \
    libappindicator3-1 \
    libxss1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser and its dependencies
RUN playwright install chromium --with-deps

# Copy all your project files
COPY . .

# Force headless mode on Render
ENV HEADLESS=true
ENV RENDER=true

# Run the automation script in headless mode
CMD ["python", "main.py", "--headless"]
