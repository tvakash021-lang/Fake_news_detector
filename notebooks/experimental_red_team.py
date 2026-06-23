import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
API_URL = "http://127.0.0.1:8000/api/v1/predict"

ADVERSARIAL_SUITE = [
    {
        "true_label": "Real News",
        "description": "True fact wrapped in sensational Fake News style formatting.",
        "text": "SHOCKING!!! ABSOLUTE PANIC!!! The Federal Reserve actually raised interest rates by 0.25% today! YOU WON'T BELIEVE IT!!!"
    },
    {
        "true_label": "Fake News",
        "description": "Completely false fact written in calm, academic journalism style.",
        "text": "Washington (AP) — In a sudden shift of diplomatic policy, the United States formally ceded the state of Florida to the Spanish government during an early morning summit, officials confirmed."
    }
]

def execute_adversarial_validation():
    logging.info("--- Initiating Adversarial Red-Team Validation ---")
    passed = 0
    for case in ADVERSARIAL_SUITE:
        payload = {"article_text": case["text"]}
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            result = response.json()
            predicted = result['prediction_label']
            
            if predicted == case["true_label"]:
                logging.info(f"✅ PASS | {case['description']}")
                passed += 1
            else:
                logging.error(f"❌ FAIL | {case['description']} | Expected: {case['true_label']} | Guessed: {predicted}")
        except Exception as e:
             logging.error(f"API Error. {e}")
             return
             
    logging.info(f"Validation Complete: {passed}/{len(ADVERSARIAL_SUITE)} passed.")

if __name__ == "__main__":
    execute_adversarial_validation()
