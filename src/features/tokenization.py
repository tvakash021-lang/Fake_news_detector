import torch
import logging
from transformers import AutoTokenizer
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class FeatureEngineer:
    def __init__(self, model_checkpoint: str = "distilbert-base-uncased", max_len: int = 256):
        self.max_len = max_len
        self.model_checkpoint = model_checkpoint
        logging.info(f"Loading '{model_checkpoint}' tokenizer. Max length capped to {max_len}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if not texts:
            raise ValueError("Input text list cannot be empty.")

        encoded_batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"]
        }
