"""
GLUE Benchmark Evaluation for Federated LoRA Models

Evaluates the trained LoRA adapters on GLUE tasks (SST-2, MRPC, QQP, etc.)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from app.model.inference import parse_dtype


def load_model_with_adapter(
    base_model: str,
    adapter_path: Optional[Path],
    device_map: str = "cuda",
    dtype: str = "auto",
):
    """Load base model with optional LoRA adapter."""
    # Load tokenizer FIRST (critical for proper embedding sizes)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path) if adapter_path and adapter_path.exists() else base_model,
            trust_remote_code=True
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    kwargs = {"trust_remote_code": True}
    torch_dtype = parse_dtype(dtype)
    if torch_dtype:
        kwargs["torch_dtype"] = torch_dtype
    if device_map.lower() != "none":
        kwargs["device_map"] = device_map
    
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    
    # Resize embeddings to match tokenizer BEFORE loading adapter
    model.resize_token_embeddings(len(tokenizer))
    
    # Now load adapter if present
    if adapter_path and adapter_path.exists():
        print(f"[ADAPTER] Loading adapter from {adapter_path}")
        
        try:
            # Load the PEFT model
            peft_model = PeftModel.from_pretrained(model, str(adapter_path))
            
            # CRITICAL FIX: Merge the adapter weights into the base model
            # Without this, the model uses the base weights and ignores the adapter!
            print(f"[ADAPTER] Merging adapter weights into base model...")
            model = peft_model.merge_and_unload()
            
            print(f"[ADAPTER] Adapter weights MERGED successfully!")
        except Exception as e:
            print(f"[ADAPTER] ERROR loading adapter: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print(f"[ADAPTER] NO ADAPTER - using base model only")
        if adapter_path:
            print(f"[ADAPTER] Adapter path does not exist: {adapter_path}")
    
    model.eval()
    return model, tokenizer


def evaluate_sst2(model, tokenizer, max_samples: int = 500):
    """Evaluate on SST-2 (Sentiment Analysis) using logit-based classification."""
    print("\n=== Evaluating on SST-2 (Sentiment) ===")
    dataset = load_dataset("glue", "sst2", split="validation")
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    labels = []
    
    # Get token IDs for "positive" and "negative"
    pos_token_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_token_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
    
    for example in tqdm(dataset, desc="SST-2"):
        text = example["sentence"]
        label = example["label"]
        
        # Improved prompt for sentiment classification
        prompt = f"Review: {text}\nSentiment:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Compare logits for "positive" vs "negative" tokens
            pos_logit = logits[pos_token_id].item()
            neg_logit = logits[neg_token_id].item()
            
            # Predict based on which has higher probability
            pred = 1 if pos_logit > neg_logit else 0
        
        predictions.append(pred)
        labels.append(label)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        "task": "sst2",
        "accuracy": accuracy,
        "f1": f1,
        "num_samples": len(labels)
    }


def evaluate_mrpc(model, tokenizer, max_samples: int = 200):
    """Evaluate on MRPC (Paraphrase Detection) using logit-based classification."""
    print("\n=== Evaluating on MRPC (Paraphrase) ===")
    dataset = load_dataset("glue", "mrpc", split="validation")
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    labels = []
    
    # Get token IDs for "yes" and "no"
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    
    for example in tqdm(dataset, desc="MRPC"):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        label = example["label"]
        
        # Improved prompt
        prompt = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre they paraphrases?"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Compare logits for "yes" vs "no"
            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            
            # Predict 1 if "yes" has higher probability
            pred = 1 if yes_logit > no_logit else 0
        
        predictions.append(pred)
        labels.append(label)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        "task": "mrpc",
        "accuracy": accuracy,
        "f1": f1,
        "num_samples": len(labels)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on GLUE benchmarks")
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--adapter", type=Path, help="Path to LoRA adapter (optional)")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"], 
                       help="GLUE tasks to evaluate")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Max samples per task (0 for all)")
    parser.add_argument("--output", type=Path, default=Path("results/glue_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model: {args.base_model}")
    model, tokenizer = load_model_with_adapter(
        args.base_model,
        args.adapter,
        args.device_map,
        args.dtype
    )
    
    results = {
        "base_model": args.base_model,
        "adapter": str(args.adapter) if args.adapter else "none",
        "tasks": {}
    }
    
    # Evaluate on selected tasks
    task_functions = {
        "sst2": evaluate_sst2,
        "mrpc": evaluate_mrpc,
    }
    
    for task in args.tasks:
        if task in task_functions:
            task_result = task_functions[task](model, tokenizer, args.max_samples)
            results["tasks"][task] = task_result
            print(f"\n{task.upper()} Results:")
            print(f"  Accuracy: {task_result['accuracy']:.4f}")
            print(f"  F1: {task_result['f1']:.4f}")
        else:
            print(f"Warning: Task '{task}' not implemented")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("GLUE BENCHMARK SUMMARY")
    print("="*60)
    for task, result in results["tasks"].items():
        print(f"{task.upper():10s} - Acc: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
