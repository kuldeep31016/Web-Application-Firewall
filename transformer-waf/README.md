# Transformer WAF

Quickstart:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run detection API
uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000 --workers 2

# Run update API
uvicorn src.api.update_api:app --host 0.0.0.0 --port 8001 --workers 1
```

Training: implement `load_training_data` and create checkpoints in `models/checkpoints/best.pt`.
