# model/summarizer.py

from transformers import BartForConditionalGeneration
from config import MODEL_NAME

class TextSummarizer:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    def get_model(self):
        return self.model
