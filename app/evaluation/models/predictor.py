import torch
from tqdm import tqdm

from app.evaluation.datasets.formatters import normalize_text


def predict(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """
    Simple wrapper around model.generate for evaluation inference.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # deterministic
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalize_text(text)
