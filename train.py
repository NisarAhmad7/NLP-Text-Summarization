# train.py

import torch
from torch.utils.data import DataLoader
from tokenizer.tokenizer import TextTokenizer
from data.dataset import SummarizationDataset
from model.summarizer import TextSummarizer
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE




def train():
    tokenizer = TextTokenizer()
    model = TextSummarizer().get_model()
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # load dataset
    dataset_hf = load_dataset("ccdv/arxiv-summarization", split="train[:1000]")

    # extract text & summary
    texts = dataset_hf["article"]
    summaries = dataset_hf["abstract"]

    # create dataset & loader
    dataset = SummarizationDataset(texts, summaries, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
