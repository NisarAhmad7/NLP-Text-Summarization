# evaluate.py

from datasets import load_dataset
from tokenizer.tokenizer import TextTokenizer
from model.summarizer import TextSummarizer
import torch

def evaluate_arxiv(max_samples=10, max_length=100):

    dataset = load_dataset("ccdv/arxiv-summarization", split=f"test[:{max_samples}]")

    tokenizer = TextTokenizer()
    model = TextSummarizer().get_model()
    model.eval()

    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for item in dataset:
            text = item["article"]
            reference = item["abstract"]

            inputs = tokenizer.encode(text)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length
            )

            generated = tokenizer.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            generated_summaries.append(generated)
            reference_summaries.append(reference)

    return generated_summaries, reference_summaries


if __name__ == "__main__":
    preds, refs = evaluate_arxiv(max_samples=5)

    for i in range(len(preds)):
        print(f"\n--- Sample {i+1} ---")
        print("Generated Summary:")
        print(preds[i])
        print("\nReference Summary:")
        print(refs[i])
