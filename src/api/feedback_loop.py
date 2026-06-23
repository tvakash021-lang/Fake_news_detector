import logging
from pydantic import BaseModel, Field
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class UserFeedback(BaseModel):
    original_text: str = Field(..., max_length=5000)
    model_prediction: str
    user_correction: str
    feedback_reason: str = Field(None, description="Optional text from user why AI failed.")

def log_feedback_for_retraining(feedback: UserFeedback):
    tgt_dir = "data/raw"
    os.makedirs(tgt_dir, exist_ok=True)
    quarantine_path = os.path.join(tgt_dir, "quarantine_feedback.jsonl")
    
    try:
        feedback_dict = feedback.model_dump()
        feedback_dict["status"] = "pending_human_review"
        
        with open(quarantine_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_dict) + "\n")
        logging.info("Feedback successfully logged for Active Learning review.")
    except Exception as e:
         logging.error(f"Failed to log feedback: {e}")
