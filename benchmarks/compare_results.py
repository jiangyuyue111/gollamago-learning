#!/usr/bin/env python3
"""Validate and compare an optimized inference result with its baseline."""

import argparse
import json
from pathlib import Path


COMPARABLE_FIELDS = (
    "target",
    "device_name",
    "torch_version",
    "maca_version",
    "seed",
    "batch_size",
    "num_input_tokens_per_sequence",
    "num_output_tokens_per_sequence",
)


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def compare(baseline: dict, candidate: dict) -> dict:
    mismatches = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in COMPARABLE_FIELDS
        if baseline.get(field) != candidate.get(field)
    }
    if mismatches:
        raise ValueError(f"Benchmark conditions differ: {json.dumps(mismatches)}")
    if baseline.get("generated_token_ids") != candidate.get("generated_token_ids"):
        raise ValueError("Generated token IDs differ; the candidate is not comparable")

    baseline_throughput = baseline["tokens_per_second"]
    candidate_throughput = candidate["tokens_per_second"]
    speedup = candidate_throughput / baseline_throughput
    return {
        "baseline_backend": baseline["backend"],
        "candidate_backend": candidate["backend"],
        "baseline_tokens_per_second": baseline_throughput,
        "candidate_tokens_per_second": candidate_throughput,
        "speedup": speedup,
        "improvement_percent": (speedup - 1) * 100,
        "output_tokens_match": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    result = compare(load_result(args.baseline), load_result(args.candidate))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result


if __name__ == "__main__":
    main()
