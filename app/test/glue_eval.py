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
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    
    model.eval()
    return model, tokenizer


def evaluate_sst2(model, tokenizer, max_samples: int = 500):
    """Evaluate on SST-2 (Sentiment Analysis)."""
    print("\n=== Evaluating on SST-2 (Sentiment) ===")
    dataset = load_dataset("glue", "sst2", split="validation")
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    labels = []
    
    for example in tqdm(dataset, desc="SST-2"):
        text = example["sentence"]
        label = example["label"]
        
        # Prompt for sentiment classification
        prompt = f"Classify the sentiment of this sentence as positive (1) or negative (0):\n{text}\nSentiment:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.1,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Parse response
        if "1" in response or "positive" in response.lower():
            pred = 1
        elif "0" in response or "negative" in response.lower():
            pred = 0
        else:
            pred = 0  # Default to negative if unclear
        
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
    """Evaluate on MRPC (Paraphrase Detection)."""
    print("\n=== Evaluating on MRPC (Paraphrase) ===")
    dataset = load_dataset("glue", "mrpc", split="validation")
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    labels = []
    
    for example in tqdm(dataset, desc="MRPC"):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        label = example["label"]
        
        prompt = f"Are these two sentences paraphrases? Answer yes (1) or no (0):\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.1,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        if "1" in response or "yes" in response.lower():
            pred = 1
        elif "0" in response or "no" in response.lower():
            pred = 0
        else:
            pred = 0
        
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
