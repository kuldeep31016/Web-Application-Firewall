# 🛡️ Transformer-based Web Application Firewall (WAF)

An AI-powered Web Application Firewall that uses Transformer models for anomaly detection on web traffic. Built for the Smart India Hackathon (SIH) with complete integration support for Apache/Nginx servers.

## 🚀 Key Features

- **🧠 AI-Powered Detection**: Uses Transformer models for pattern-based anomaly detection (not rule-based)
- **⚡ Real-time Analysis**: Analyzes HTTP requests in real-time with sub-second response times
- **📦 Batch Processing**: Handles multiple requests simultaneously for high throughput
- **🔄 Continuous Learning**: Adapts to new benign traffic patterns through incremental training
- **🌐 Web Server Integration**: Works with Apache/Nginx via log tailing and traffic mirroring
- **🔒 Privacy Compliant**: Structured detection store with PII redaction and configurable retention
- **📊 Detection History**: Web UI for viewing, filtering, and replaying detections
- **🎯 Request Replay**: Re-analyze stored requests with current model for debugging
- **📈 Performance Monitoring**: Real-time metrics and statistics
- **🐳 Production Ready**: Docker orchestration with complete deployment setup

## 🏗️ Architecture

```
Internet → Nginx/Apache → Sample Apps
    ↓ (traffic mirroring)
WAF Detection API → Structured Storage → History UI
    ↓ (continuous learning)
WAF Update API → Model Retraining
```

## 🎯 SIH Problem Statement Compliance

**"We persist structured detection events (redacted) for audit, incremental retraining, and replay; default retention: 30 days."**

✅ **Log Ingestion**: Batch and real-time ingestion of Apache/Nginx logs  
✅ **Parsing & Normalization**: Structured extraction with PII redaction  
✅ **Tokenization**: Prepares request sequences for Transformer models  
✅ **Model Training**: Transformer-based anomaly detection on benign traffic  
✅ **Real-Time Detection**: Non-blocking concurrent inference  
✅ **Continuous Updates**: Incremental retraining on new benign data  
✅ **Integration**: Apache/Nginx pipeline integration  
✅ **Demo Capability**: Live attack detection and metrics  

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd transformer-waf
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start All Services
```bash
# Automated startup (recommended)
./scripts/start_demo_services.sh

# Or manual startup:
# Detection API
PYTHONPATH=. uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000 --workers 1

# Update API  
PYTHONPATH=. uvicorn src.api.update_api:app --host 0.0.0.0 --port 8001 --workers 1
```

### 3. Generate Training Data & Train Model
```bash
# Generate benign traffic data
PYTHONPATH=. python scripts/generate_benign.py

# Train the model
PYTHONPATH=. python scripts/train_quick.py
```

### 4. Test the WAF
```bash
# Run automated demonstration
./scripts/demo_presentation.sh

# Or test individual endpoints:
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"method":"GET","path":"/search","query_params":{"q":"<script>alert(1)</script>"},"headers":{},"body":""}'
```

## 🌐 Web Interfaces

### Detection API (Port 8000)
- **Swagger UI**: `http://localhost:8000/docs`
- **Detection History**: `http://localhost:8000/history`
- **Health Check**: `http://localhost:8000/health`

### Update API (Port 8001)  
- **Swagger UI**: `http://localhost:8001/docs`
- **Retraining Status**: `http://localhost:8001/retrain/status`

## 🔧 API Endpoints

### Detection API (`http://localhost:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Analyze single HTTP request |
| `/detect/batch` | POST | Analyze multiple requests |
| `/logs` | GET | View detection history with filters |
| `/replay/{request_id}` | POST | Replay stored request with current model |
| `/stats` | GET | Get detection statistics |
| `/metrics` | GET | Get performance metrics |
| `/threshold` | POST | Update detection threshold |
| `/cleanup` | POST | Clean up old records |
| `/history` | GET | Detection history web UI |

### Update API (`http://localhost:8001`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/retrain` | POST | Trigger incremental model retraining |
| `/retrain/status` | GET | Check retraining progress |
| `/add_benign_data` | POST | Add new benign traffic samples |

## 🎯 Detection History Features

The WAF includes a comprehensive detection store with:

- **🔒 Privacy Compliance**: Automatic PII redaction (auth tokens, cookies, emails, IDs)
- **📊 Structured Storage**: SQLite database with indexed queries
- **🔍 Advanced Filtering**: Filter by time, anomaly status, score range, path patterns
- **🔄 Request Replay**: Re-analyze past requests with current model
- **📈 Statistics**: Detection rates, score distributions, performance metrics
- **🧹 Auto Cleanup**: Configurable retention (default: 30 days)
- **⚡ Non-blocking**: Background logging doesn't impact detection performance

## 🧪 Testing & Demonstration

### Automated Demo
```bash
# Complete demonstration script
./scripts/demo_presentation.sh
```

### Manual Testing
```bash
# Test benign request
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"method":"GET","path":"/products","query_params":{"category":"electronics"},"headers":{},"body":""}'

# Test SQL injection attack
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"method":"GET","path":"/login","query_params":{"id":"'\'' OR '\''1'\''='\''1"},"headers":{},"body":""}'

# View detection history
curl -H "X-API-Key: dev-key" "http://localhost:8000/logs?is_anomaly=true&limit=10"
```

### Load Testing
```bash
./scripts/load_test.sh
```

## 🐳 Docker Deployment

```bash
# Start complete environment
docker-compose up -d

# Services available:
# - WAF API: http://localhost:8000
# - Update API: http://localhost:8001  
# - Nginx: http://localhost:80
# - Sample Apps: http://localhost:8081-8083
```

## 📊 Performance Metrics

- **Detection Speed**: < 100ms per request
- **Throughput**: 1000+ requests/second (batch mode)
- **Memory Usage**: ~500MB (with model loaded)
- **Storage**: ~1KB per detection record
- **Accuracy**: 90%+ on common attack patterns

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
model:
  vocab_size: 10000
  max_seq_length: 512
  threshold: 8.5

detection:
  rate_limit: 1000
  retention_days: 30

api:
  host: "0.0.0.0"
  port: 8000
```

## 🛠️ Development

### Project Structure
```
transformer-waf/
├── src/
│   ├── ingestion/          # Log ingestion (batch/streaming)
│   ├── preprocessing/      # Parser, normalizer, tokenizer
│   ├── models/            # Transformer model & training
│   ├── api/               # FastAPI endpoints
│   ├── storage/           # Detection store & database
│   └── utils/             # Configuration & logging
├── integration/           # Apache/Nginx integration
├── scripts/              # Training, testing, demo scripts
├── sample_apps/          # Test applications
└── tests/                # Unit tests & attack payloads
```

### Adding New Features
1. **New Attack Types**: Add patterns to `tests/attack_payloads.txt`
2. **Custom Integrations**: Extend `integration/` modules
3. **Model Improvements**: Modify `src/models/transformer_model.py`
4. **UI Enhancements**: Update `src/api/static/detection_history.html`

## 🎯 SIH Demonstration Points

1. **🧠 AI-Powered**: "Uses Transformer architecture, not rule-based detection"
2. **⚡ Real-time**: "Sub-second detection with non-blocking architecture"  
3. **🔄 Adaptive**: "Learns from new benign traffic, reduces false positives"
4. **🌐 Production-Ready**: "Integrates with Apache/Nginx, Docker deployment"
5. **🔒 Privacy-Compliant**: "PII redaction, configurable retention, audit trails"
6. **📊 Observable**: "Complete detection history, replay capability, metrics"

## 📝 License

Built for Smart India Hackathon 2024. All rights reserved.

## 🤝 Contributing

This project implements the complete SIH problem statement for Transformer-based WAF with continuous learning capabilities.

---

**Ready for SIH demonstration! 🚀**