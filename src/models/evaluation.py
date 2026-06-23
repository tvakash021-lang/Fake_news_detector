import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def evaluate_production_model(y_true: np.ndarray, y_pred: np.ndarray, min_recall: float = 0.90):
    logging.info("--- Initiating Strict Model Evaluation ---")
    
    target_names = ['Fake News', 'Real News']
    report_str = classification_report(y_true, y_pred, target_names=target_names)
    logging.info(f"\nClassification Report:\n{report_str}")
    
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"\nConfusion Matrix:\n{cm}")
    
    fake_news_recall = recall_score(y_true, y_pred, pos_label=0)
    global_f1 = f1_score(y_true, y_pred, average='weighted')
    
    logging.info(f"Fake News Recall: {fake_news_recall:.4f}")
    logging.info(f"Global F1 Score:  {global_f1:.4f}")
    
    if fake_news_recall < min_recall:
        logging.error(f"DEPLOYMENT BLOCKED: Fake News Recall is below required SLA ({min_recall:.4f}).")
        raise ValueError("Model failed critical recall threshold validation.")
        
    logging.info("✅ Model passed all evaluation SLAs. Ready for API integration.")
    
    metrics = {
        "fake_news_recall": float(fake_news_recall),
        "global_f1": float(global_f1),
        "confusion_matrix": cm.tolist()
    }
    with open("data/processed/latest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
