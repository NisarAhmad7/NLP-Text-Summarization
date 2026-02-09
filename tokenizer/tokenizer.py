
# tokenizer/tokenizer.py

from transformers import BartTokenizer
from config import MAX_INPUT_LENGTH, MAX_SUMMARY_LENGTH, MODEL_NAME

class TextTokenizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

    def encode(self, text, is_summary=False):
        max_len = MAX_SUMMARY_LENGTH if is_summary else MAX_INPUT_LENGTH
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
