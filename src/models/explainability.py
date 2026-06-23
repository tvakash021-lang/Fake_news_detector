import torch
import logging
from lime.lime_text import LimeTextExplainer
from typing import List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class LimeExplainabilityModule:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

    def predictor_function(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return probs.cpu().numpy()

    def generate_explanation(self, text: str, num_features: int = 10) -> List[Tuple[str, float]]:
        logging.info("Generating LIME explanation...")
        exp = self.explainer.explain_instance(
            text, 
            self.predictor_function, 
            num_features=num_features,
            num_samples=100 
        )
        return exp.as_list()
