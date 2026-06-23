import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class ProjectConfig(BaseSettings):
    PROJECT_NAME: str = "Fake News Detection Engine"
    API_VERSION: str = "v1"
    
    MAX_LATENCY_MS: int = Field(default=250, description="Absolute maximum allowable inference latency in milliseconds.")
    TARGET_THROUGHPUT_RPS: int = Field(default=50, description="Target Requests Per Second for baseline deployment.")
    
    MIN_F1_SCORE: float = Field(default=0.85, description="Minimum F1 score required for a model to pass CI/CD.")
    MIN_RECALL: float = Field(default=0.90, description="Minimum Recall; prefer false positives over fake news spread.")
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

config = ProjectConfig()
