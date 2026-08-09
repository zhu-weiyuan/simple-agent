@echo off
cd /d C:\Users\Administrator\.openclaw\workspace1\simple-agent
set OPENAI_API_KEY=sk-local
set OPENAI_BASE_URL=http://localhost:8080/v1
set OPENAI_MODEL=D:\download\KAT-Coder-V2.5-Dev-APEX-I-Quality.gguf
set API_KEYS=test-key-123
set JWT_SECRET=test-jwt
set RATE_LIMIT_REQUESTS=500
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
