import pandas as pd
import logging
import os
import requests
import io
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def download_and_format_dataset():
    """
    Downloads a recognized benchmarking Fake News dataset, normalizes the labels 
    to numeric [Fake=0, Real=1], and drops it into our immutable data/raw directory.
    """
    # Utilizing an open-source Fake vs Real news dataset commonly used in academia
    url = "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip"
    logging.info(f"Downloading raw dataset buffer from: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Load CSV buffer into Memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('fake_or_real_news.csv') as f:
                df = pd.read_csv(f)
        
        # Standardize to API boundary expectations
        # Original schema: 'title', 'text', 'label' -> (where label is 'FAKE' or 'REAL')
        if 'label' in df.columns and 'text' in df.columns:
            # Drop nulls and select columns
            df = df.dropna(subset=['text', 'label'])
            df = df[['title', 'text', 'label']].copy()
            
            # Map strings to standard integers for the Transformer
            df['mapped_label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
            
            # Defensive check
            if df['mapped_label'].isnull().any():
                 logging.warning("Unrecognized labels found during mapping phase.")
                 df = df.dropna(subset=['mapped_label'])
                 
            # Ensure strictly integers
            df['mapped_label'] = df['mapped_label'].astype(int)
            
            # Save artifact
            os.makedirs("data/raw", exist_ok=True)
            output_path = "data/raw/primary_dataset.csv"
            
            df.to_csv(output_path, index=False)
            logging.info(f"✅ Ingestion complete. Hydrated {len(df)} records safely to {output_path}")
        else:
            logging.error("Dataset schema violation. Expected ['text', 'label'] columns.")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Network failure during dataset retrieval: {e}")
    except Exception as e:
        logging.error(f"Panic during data frame formatting: {e}")

if __name__ == "__main__":
    download_and_format_dataset()
