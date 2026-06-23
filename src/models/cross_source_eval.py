import pandas as pd
import logging
from sklearn.metrics import f1_score
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def evaluate_cross_source_variance(predictions_df: pd.DataFrame) -> Dict[str, float]:
    if 'source_domain' not in predictions_df.columns:
         logging.error("Missing 'source_domain' metadata.")
         return {}
         
    domains = predictions_df['source_domain'].unique()
    domain_scores = {}
    
    logging.info("--- Stratified Cross-Source F1 Breakdown ---")
    for domain in domains:
        subset = predictions_df[predictions_df['source_domain'] == domain]
        score = f1_score(subset['true_label'], subset['pred_label'], zero_division=0)
        domain_scores[domain] = score
        logging.info(f"Domain: {domain.ljust(20)} | F1: {score:.4f}")

    scores = list(domain_scores.values())
    variance_spread = max(scores) - min(scores)
    logging.info(f"Cross-Source Variance Spread: {variance_spread:.4f}")
    
    if variance_spread > 0.15:
        logging.warning("GENERALIZATION FAILURE: Model performance fluctuates wildly depending on the source.")
        
    return domain_scores
