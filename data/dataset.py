# data/dataset.py

import torch
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_enc = self.tokenizer.encode(self.texts[idx])
        target_enc = self.tokenizer.encode(self.summaries[idx], is_summary=True)

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }
