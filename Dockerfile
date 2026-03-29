FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HEADLESS=true
ENV RENDER=true
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--headless"]
