"""
GLUE Benchmark Evaluation for Base and LoRA Fine-Tuned Models

Evaluates the base Qwen 1.8B model and all LoRA adapters on GLUE tasks (SST-2, MRPC).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from app.model.inference import parse_dtype


def load_model_with_adapter(
    base_model: str,
    adapter_path: Optional[str],
    device_map: str = "cuda",
    dtype: str = "auto",
):
    """Load base model with optional LoRA adapter."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if adapter_path else base_model,
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
    
    model.resize_token_embeddings(len(tokenizer))
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
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
        
        # Prompt for paraphrase detection
        prompt = f"Are these two sentences paraphrases? Answer yes (1) or no (0):\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nParaphrase:"
        
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
        
        # Parse response
        if "1" in response or "yes" in response.lower():
            pred = 1
        elif "0" in response or "no" in response.lower():
            pred = 0
        else:
            pred = 0  # Default to no if unclear
        
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


def run_glue_evaluation(base_model: str, adapter: Optional[str], tasks: List[str], max_samples: int, device_map: str, dtype: str):
    """Run GLUE evaluation for a model configuration."""
    print(f"Loading model: {base_model}")
    if adapter:
        print(f"With adapter: {adapter}")
    model, tokenizer = load_model_with_adapter(base_model, adapter, device_map, dtype)
    
    results = {}
    for task in tasks:
        if task == "sst2":
            results[task] = evaluate_sst2(model, tokenizer, max_samples)
        elif task == "mrpc":
            results[task] = evaluate_mrpc(model, tokenizer, max_samples)
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate base and LoRA models on GLUE benchmarks")
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--config", type=Path, default=Path("app/model/dept_lora_base.json"))
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/glue_comparison_results.json"))
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    all_results = {}
    
    # Evaluate base model
    print("Evaluating base model...")
    all_results["base"] = run_glue_evaluation(
        args.base_model, None, args.tasks, args.max_samples, args.device_map, args.dtype
    )
    
    # Evaluate each department adapter
    for dept, adapters in config.items():
        print(f"\nEvaluating {dept} adapter...")
        # Assuming one adapter per dept
        adapter = adapters[0]
        all_results[dept] = run_glue_evaluation(
            args.base_model, adapter, args.tasks, args.max_samples, args.device_map, args.dtype
        )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "config": str(args.config),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary
    print("""
============================================================
GLUE BENCHMARK COMPARISON SUMMARY
============================================================
""")
    for model_name, tasks in all_results.items():
        print(f"\n{model_name.upper()}:")
        for task, metrics in tasks.items():
            print(f"  {task.upper()} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
