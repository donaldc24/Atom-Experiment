import argparse

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_SENTENCE = "BERT turns this sentence into contextual token embeddings."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass one sentence through BERT.")
    parser.add_argument(
        "sentence",
        nargs="?",
        default=DEFAULT_SENTENCE,
        help="Sentence to encode (a built-in example is used by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Loading {MODEL_NAME} (downloads on the first run)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(args.sentence, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Sentence: {args.sentence}")
    print(f"Tokens: {tokens}")
    print(f"Input IDs shape: {tuple(inputs['input_ids'].shape)}")
    print(f"BERT output shape: {tuple(outputs.last_hidden_state.shape)}")
    print("Done: the sentence passed through BERT successfully.")

    sample = load_dataset(
        "nyu-mll/glue",
        "sst2",
        split="train[:2]",
    )

    print("\nSST-2 samples:")
    for row in sample:
        print(row)


if __name__ == "__main__":
    main()
