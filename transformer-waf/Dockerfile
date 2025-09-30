FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WAF_CONFIG_PATH=/app/config.yaml
EXPOSE 8000

CMD ["uvicorn", "src.api.detection_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


