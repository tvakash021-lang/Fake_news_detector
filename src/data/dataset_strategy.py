import pandas as pd
import logging
from datetime import datetime
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def analyze_dataset_viability(df: pd.DataFrame, text_col: str, label_col: str, date_col: str = None) -> Dict[str, float]:
    stats = {}
    
    class_counts = df[label_col].value_counts(normalize=True)
    majority_class_ratio = class_counts.max()
    stats['majority_class_ratio'] = majority_class_ratio
    
    logging.info(f"Class Distribution:\n{class_counts.to_string()}")
    if majority_class_ratio > 0.70:
        logging.warning("SEVERE BIAS DETECTED: Dataset is heavily imbalanced.")
        
    df['text_length'] = df[text_col].apply(lambda x: len(str(x).split()))
    avg_length_by_class = df.groupby(label_col)['text_length'].mean()
    stats['length_ratio_difference'] = abs(avg_length_by_class.iloc[0] - avg_length_by_class.iloc[-1])
    
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        most_recent_date = df[date_col].max()
        oldest_date = df[date_col].min()
        
        current_date = pd.to_datetime(datetime.now())
        months_since_newest = (current_date.year - most_recent_date.year) * 12 + (current_date.month - most_recent_date.month)
        
        stats['months_since_newest_data'] = months_since_newest
        stats['data_timespan_days'] = (most_recent_date - oldest_date).days
        
        if months_since_newest > 24:
            logging.error("RECENCY FAILURE: Newest data is over 2 years old.")
            
    return stats
