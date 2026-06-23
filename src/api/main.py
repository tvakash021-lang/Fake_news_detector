import time
import logging
from fastapi import FastAPI, HTTPException
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.api.schemas import ArticleRequest, PredictionResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

app = FastAPI(title="Fake News Detection API", version="1.0.0")
ml_components = {}

@app.on_event("startup")
async def load_model_on_startup():
    model_id = "therealcyberlord/fake-news-classification-distilbert"
    logging.info(f"API STARTUP: Booting ML architecture from Hub model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        ml_components['tokenizer'] = tokenizer
        ml_components['model'] = model
        ml_components['device'] = device
        logging.info("ML Engine safely loaded from HuggingFace Hub.")
    except Exception as e:
        logging.error(f"Failed to load model from HuggingFace Hub. Error: {e}")

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_fake_news(request: ArticleRequest):
    start_time = time.time()
    try:
        clean_text = request.article_text[:256] 
        inputs = ml_components['tokenizer'](
            clean_text, return_tensors="pt", truncation=True, max_length=256
        ).to(ml_components['device'])
        
        with torch.no_grad():
            outputs = ml_components['model'](**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
        labels = ["Fake News", "Real News"]
        prediction = labels[predicted_idx.item()]
        process_time_ms = int((time.time() - start_time) * 1000)
        
        return PredictionResponse(
            prediction_label=prediction,
            confidence_score=confidence.item(),
            processing_time_ms=process_time_ms
        )
    except Exception as e:
        logging.error(f"Inference crash: {e}")
        raise HTTPException(status_code=500, detail="Internal ML processing error.")
