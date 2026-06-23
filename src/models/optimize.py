import torch
import logging
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def optimize_model_for_production(model_path: str, output_path: str):
    logging.info(f"Loading heavy PyTorch model from {model_path}...")
    try:
        ort_model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        ort_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logging.info(f"ONNX model saved to {output_path}.")
        
        size_mb = os.path.getsize(os.path.join(output_path, "model.onnx")) / (1024 * 1024)
        logging.info(f"Optimized Benchmark: ONNX Size is ~{size_mb:.1f} MB.")
    except Exception as e:
        logging.error(f"Optimization failed: {e}")

if __name__ == "__main__":
    pytorch_dir = "./models/production_model"
    onnx_dir = "./models/optimized_onnx_model"
    if os.path.exists(pytorch_dir):
        optimize_model_for_production(pytorch_dir, onnx_dir)
