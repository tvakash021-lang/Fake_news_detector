import re
import html
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.twitter_handles = re.compile(r'@\w+')
        self.spaces_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s.,!\'"-]')

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = text.lower()
        text = self.url_pattern.sub('', text)
        text = self.html_pattern.sub('', text)
        text = self.twitter_handles.sub('', text)
        text = self.special_chars_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text).strip()
        return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]
