# 🕵️‍♂️ Detection Engine: Production Implementation

## 🚀 Project Overview
An end-to-end, async-optimized Fake News classification API powered by a fine-tuned Deep Learning Transformer (DistilBERT). Designed for high-concurrency cloud environments with strict latency SLA bounds.

## 🛠 Architecture Deliverables
1. **Model Core:** INT8 Quantized ONNX Transformer (Optimized for CPU inference speeds <100ms).
2. **API Layer:** Asynchronous FastAPI backend decoupling the ML logic from network I/O.
3. **Client UI:** Streamlit dashboard utilizing HTTP Request boundaries.
4. **XAI:** Live LIME (Local Interpretable Model-agnostic Explanations) integration.

## 🐳 Quickstart 
To run via Python locally:
```bash
make setup
make run-all
```

To run via Docker:
```bash
docker-compose up --build
```
*   **UI Dashboard:** `http://localhost:8501`
*   **API Docs:** `http://localhost:8000/docs`
