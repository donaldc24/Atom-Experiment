# PyTorch BERT demo

A minimal Python project that:

1. imports PyTorch,
2. downloads the pretrained `prajjwal1/bert-tiny` tokenizer and model,
3. passes one sentence through BERT, and
4. downloads and prints two SST-2 training examples.

The model is downloaded from Hugging Face on the first run and cached locally for later runs.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
bert-demo
```

You can also supply your own sentence:

```powershell
bert-demo "PyTorch makes tensor computation straightforward."
```

The final tensor shape is `(batch size, token count, hidden size)`. For BERT Tiny, the hidden
size is 128.
