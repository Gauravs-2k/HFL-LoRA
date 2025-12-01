import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.evaluation.datasets.department_loader import load_department_dataset
from app.evaluation.models.loader import load_model
from app.evaluation.runners.eval_single import evaluate_single_model
from app.model.inference import parse_dtype
from app.utils.config import settings


TEST_FILES = {
	"finance": "FINANCE_test.jsonl",
	"hr": "HR_test.jsonl",
	"engineering": "ENGINEERING_test.jsonl",
	"it_support": "IT_SUPPORT_test.jsonl",
}


def resolve_base_model(value):
    if value:
        return value
    # Default to Qwen1.5-1.8B-Chat for HF-based evaluation
    return "Qwen/Qwen1.5-1.8B-Chat"
def dataset_path(root, department):
	filename = TEST_FILES.get(department)
	if not filename:
		raise ValueError(f"Unsupported department {department}")
	return Path(root) / filename


def _hf_kwargs():
	kwargs = {"trust_remote_code": True}
	if settings.HF_TOKEN:
		kwargs["token"] = settings.HF_TOKEN
	return kwargs


def load_remote_full(model_id, dtype, device_map):
	tokenizer = AutoTokenizer.from_pretrained(model_id, **_hf_kwargs())
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model_kwargs = {"trust_remote_code": True}
	torch_dtype = parse_dtype(dtype)
	if torch_dtype:
		model_kwargs["torch_dtype"] = torch_dtype
	if device_map.lower() != "none":
		model_kwargs["device_map"] = device_map
	if settings.HF_TOKEN:
		model_kwargs["token"] = settings.HF_TOKEN
	model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
	model.resize_token_embeddings(len(tokenizer))
	model.eval()
	return tokenizer, model


def load_remote_adapter(base_model, adapter_id, dtype, device_map):
	try:
		tokenizer = AutoTokenizer.from_pretrained(adapter_id, **_hf_kwargs())
	except Exception:
		tokenizer = AutoTokenizer.from_pretrained(base_model, **_hf_kwargs())
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model_kwargs = {"trust_remote_code": True}
	torch_dtype = parse_dtype(dtype)
	if torch_dtype:
		model_kwargs["torch_dtype"] = torch_dtype
	if device_map.lower() != "none":
		model_kwargs["device_map"] = device_map
	if settings.HF_TOKEN:
		model_kwargs["token"] = settings.HF_TOKEN
	model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
	model.resize_token_embeddings(len(tokenizer))
	model = PeftModel.from_pretrained(model, adapter_id, is_trainable=False)
	model.eval()
	return tokenizer, model


def load_pair(base_model, candidate, dtype, device_map):
	if not candidate:
		return load_model(base_model, None, dtype, device_map)
	path = Path(candidate)
	if path.exists():
		return load_model(base_model, str(path), dtype, device_map)
	try:
		return load_remote_adapter(base_model, candidate, dtype, device_map)
	except Exception as adapter_err:
		try:
			return load_remote_full(candidate, dtype, device_map)
		except Exception as full_err:
			raise RuntimeError(
				f"Failed to load {candidate} as adapter ({adapter_err}) and as full model ({full_err})"
			)


def summarize_result(result):
	summary = {"accuracy": result.get("accuracy", 0.0)}
	return summary


def evaluate_model(base_model, candidate, dataset, dtype, device_map):
	tokenizer, model = load_pair(base_model, candidate, dtype, device_map)
	res = evaluate_single_model(model, tokenizer, dataset)
	summary = summarize_result(res)
	del model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return summary


def run():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-model")
	parser.add_argument("--config", default="app/model/dept_lora_base.json")
	parser.add_argument("--dataset-root", default="app/dataset/test")
	parser.add_argument("--departments", nargs="*")
	parser.add_argument("--max-samples", type=int)
	parser.add_argument("--dtype", default="auto")
	parser.add_argument("--device-map", default="auto")
	parser.add_argument("--output", default="results/dept_specific_compare.json")
	args = parser.parse_args()

	base_model = resolve_base_model(args.base_model)
	config_path = Path(args.config)
	with open(config_path, "r", encoding="utf-8") as handle:
		config = json.load(handle)
	selected = args.departments or list(config.keys())
	results = []

	for dept in selected:
		try:
			data_path = dataset_path(args.dataset_root, dept)
		except ValueError as exc:
			print(f"Skipping {dept}: {exc}")
			continue
		if not data_path.exists():
			print(f"Dataset missing for {dept}: {data_path}")
			continue
		dataset = load_department_dataset(str(data_path), max_samples=args.max_samples)
		if not dataset:
			print(f"Dataset empty for {dept}: {data_path}")
			continue
		print(f"\n{dept.upper()} :: samples={len(dataset)}")
		try:
			base_summary = evaluate_model(base_model, None, dataset, args.dtype, args.device_map)
			print(f"  base ({base_model}) accuracy: {base_summary['accuracy']:.4f}")
		except Exception as exc:
			print(f"  Failed base evaluation: {exc}")
			base_summary = {"error": str(exc), "accuracy": 0.0}
		dept_entry = {
			"department": dept,
			"dataset": str(data_path),
			"num_samples": len(dataset),
			"base": base_summary,
			"models": [],
		}
		for candidate in config.get(dept, []):
			print(f"  evaluating {candidate}...")
			try:
				model_summary = evaluate_model(base_model, candidate, dataset, args.dtype, args.device_map)
				model_summary["id"] = candidate
				if "accuracy" in base_summary:
					model_summary["delta_vs_base"] = model_summary["accuracy"] - base_summary.get("accuracy", 0.0)
				print(
					f"    accuracy: {model_summary['accuracy']:.4f} (Δ {model_summary.get('delta_vs_base', 0.0):+.4f})"
				)
			except Exception as exc:
				model_summary = {"id": candidate, "error": str(exc)}
				print(f"    failed: {exc}")
			dept_entry["models"].append(model_summary)
		results.append(dept_entry)

	output_data = {
		"base_model": base_model,
		"config": str(config_path),
		"results": results,
	}
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as handle:
		json.dump(output_data, handle, indent=2)
	print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
	run()


# source env/bin/activate && PYTHONPATH=$PWD python app/test/dept_specific_compare.py --device-map auto --dtype auto