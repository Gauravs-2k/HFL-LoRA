from typing import Optional

from app.model.inference import _get_model_and_tokenizer


def load_model(
    base_model: str,
    lora_dir: Optional[str] = None,
    dtype: str = "auto",
    device_map: str = "auto",
):
    """
    Thin wrapper around inference._get_model_and_tokenizer.

    Returns (tokenizer, model) so callers can use model.generate(...)
    directly in evaluation.
    """
    peft_dir = lora_dir or ""  # inference code expects a string, not None
    model, tokenizer = _get_model_and_tokenizer(base_model, peft_dir, dtype, device_map)
    return tokenizer, model


def locate_federated_round_adapter(department: str, round_num: int) -> Optional[str]:
    """
    Assumes federated adapters are stored like:
        federated_round_outputs/<department>_round_<num>/
    """
    import os

    path = f"federated_round_outputs/{department}_round_{round_num}"
    return path if os.path.isdir(path) else None


def locate_department_lora(department: str) -> Optional[str]:
    """
    Assumes centralized LoRA adapters are stored like:
        qwen_dept_lora_<department>/
    """
    import os

    path = f"qwen_dept_lora_{department}"
    return path if os.path.isdir(path) else None
