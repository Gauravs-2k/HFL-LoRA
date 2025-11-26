import argparse
from app.evaluation.datasets.department_loader import load_department_dataset
from app.evaluation.runners.eval_compare import compare_models
from app.evaluation.runners.eval_rounds import evaluate_rounds
from app.utils.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--department", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rounds", nargs="+", type=int, default=[0])
    parser.add_argument("--mode", choices=["compare", "rounds"], default="compare")
    args = parser.parse_args()

    dataset = load_department_dataset(args.dataset)
    base_model = settings.LLM_MODEL

    if args.mode == "compare":
        res = compare_models(base_model, args.department, dataset, args.rounds)
        print(res)
    else:
        res = evaluate_rounds(base_model, args.department, dataset, args.rounds)
        print(res)


if __name__ == "__main__":
    main()
