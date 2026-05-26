import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def run_llm_evaluation(prediction_file, dataset_file, output_file, max_workers):
    from llm_evaluate_predictions import evaluate_single_item, load_jsonl

    print(f"Loading predictions from {prediction_file}...")
    predictions = load_jsonl(prediction_file)
    if predictions is None:
        return None

    print(f"Loading dataset from {dataset_file}...")
    dataset = load_jsonl(dataset_file)
    if dataset is None:
        return None

    ground_truth_map = {item["id"]: item for item in dataset}

    tasks = []
    for pred in predictions:
        item_id = pred.get("id")
        if not item_id:
            print(
                f"Warning: Missing 'id' in prediction, skipping record: {pred}",
                file=sys.stderr,
            )
            continue
        if item_id in ground_truth_map:
            tasks.append((pred, ground_truth_map[item_id]))
        else:
            print(
                f"Warning: ID '{item_id}' not found in dataset, skipping this prediction.",
                file=sys.stderr,
            )

    total_items = len(tasks)
    if total_items == 0:
        print("No matching items to evaluate. Exiting.")
        return None

    print(
        f"Will evaluate {total_items} records (using up to {max_workers} worker threads)...\n"
    )

    correct_count = 0
    total_judged = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(lambda task: evaluate_single_item(task), tasks)

            for result in tqdm(
                results_iterator,
                total=total_items,
                desc="Evaluating predictions",
            ):
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
                if result.get("judgement") == "True":
                    correct_count += 1
                if result.get("judgement") in ["True", "False"]:
                    total_judged += 1

    accuracy = (correct_count / total_judged) if total_judged > 0 else 0.0

    print(f"\nLLM evaluation completed, results saved to: {output_file}")
    print("\n--- LLM Evaluation Summary ---")
    print(f"Number of correct judgements: {correct_count}")
    print(f"Total number of valid judgements: {total_judged}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------")

    return {
        "correct_count": correct_count,
        "judged_count": total_judged,
        "accuracy": accuracy,
        "output_file": output_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM evaluation first, then run string-based evaluation on the "
            "generated LLM evaluation result file."
        )
    )
    parser.add_argument(
        "--prediction_file",
        required=True,
        help="Path to JSONL file containing model predictions.",
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to original dataset JSONL file containing ground truth answers.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to JSONL file to save LLM evaluation results.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker threads for parallel processing (default 16).",
    )
    args = parser.parse_args()

    llm_result = run_llm_evaluation(
        prediction_file=args.prediction_file,
        dataset_file=args.dataset_file,
        output_file=args.output_file,
        max_workers=args.max_workers,
    )
    if llm_result is None:
        return

    print("\nRunning string-based evaluation...")
    try:
        from evaluation_script import evaluate as evaluate_string_metrics
    except ModuleNotFoundError as exc:
        print(
            f"String-based evaluation cannot start because a dependency is missing: {exc}",
            file=sys.stderr,
        )
        return

    string_result = evaluate_string_metrics(args.dataset_file, args.output_file)
    if string_result is None:
        print("String-based evaluation failed.")
        return

    print("\nString-based evaluation completed.")

    downstream = string_result.get("downstream", {})
    print("\n--- Evaluation Summary ---")
    print(f"Cover EM: {downstream.get('cover_em', 0.0) * 100:.2f}%")
    print(f"F1: {downstream.get('f1', 0.0) * 100:.2f}%")
    #print(f"LLM Accuracy: {downstream.get('accuracy', 0.0) * 100:.2f}%")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
