from pydantic import BaseModel, Field, validator
from typing import Dict, Optional

class ArticleRequest(BaseModel):
    article_text: str = Field(..., min_length=50, max_length=5000, description="Raw text of the news article.")
    source_url: Optional[str] = Field(None, description="Optional metadata for future origin tracking.")

    @validator('article_text')
    def text_must_not_be_whitespace(cls, v):
        if not v.strip():
            raise ValueError('Article text cannot be empty or only whitespace.')
        return v

class PredictionResponse(BaseModel):
    prediction_label: str = Field(..., description="'Fake News' or 'Real News'")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int = Field(..., description="Time taken to process to monitor SLAs.")
    lime_explanation: Optional[Dict[str, float]] = Field(None, description="Feature importance map.")
