"""
Build train/dev PileNER 391 JSONL with hierarchical guideline prompts.

The script samples up to N positive and M negative examples per entity type
from the no-split PileNER 391 JSONL. Dev examples are selected after train
examples, and any dev candidate whose input text exactly matches a selected
train input is skipped.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_PATH = "data/pileNER/pileNER_391_all.jsonl"
DEFAULT_GUIDELINES_PATH = "data/pileNER/pileNER_391_hierarchical_guidelines.json"
DEFAULT_OUTPUT_DIR = "data/pileNER/pileNER_391_5pos_5neg_with_guidelines"

GUIDELINE_FIELDS = [
    "entity_type",
    "definition",
    "include_rules",
    "exclude_rules",
    "boundary_rules",
    "confusable_type_rules",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PileNER 391 train/dev JSONL with hierarchical guideline prompts."
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="Input PileNER 391 JSONL path.")
    parser.add_argument(
        "--guidelines_path",
        default=DEFAULT_GUIDELINES_PATH,
        help="Input hierarchical guidelines JSON path.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for train/dev JSONL.")
    parser.add_argument("--n_pos", type=int, default=5, help="Positive examples per entity type per split.")
    parser.add_argument("--n_neg", type=int, default=5, help="Negative examples per entity type per split.")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducible sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite train.jsonl/dev.jsonl if present.")
    return parser.parse_args()


def parse_output_list(sample, input_path, line_number):
    try:
        output = json.loads(sample["output"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid output JSON list at {input_path}:{line_number}") from exc

    if not isinstance(output, list):
        raise ValueError(f"Expected output JSON list at {input_path}:{line_number}")
    return output


def load_samples(input_path):
    samples = []
    with open(input_path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}") from exc

            for field in ["doc_tag_pairID", "tagName", "input", "output"]:
                if field not in sample:
                    raise ValueError(f"Missing field {field!r} at {input_path}:{line_number}")

            output = parse_output_list(sample, input_path, line_number)
            samples.append((sample, output))
    return samples


def parse_guideline(ne_type, guideline):
    if not isinstance(guideline, dict):
        raise ValueError(f"Guideline for {ne_type!r} must be an object")

    missing_fields = [field for field in GUIDELINE_FIELDS if field not in guideline]
    if missing_fields:
        raise ValueError(f"Missing guideline fields for {ne_type!r}: {missing_fields}")

    if not isinstance(guideline["entity_type"], str) or not guideline["entity_type"].strip():
        raise ValueError(f"Guideline entity_type for {ne_type!r} must be a non-empty string")
    if not isinstance(guideline["definition"], str) or not guideline["definition"].strip():
        raise ValueError(f"Guideline definition for {ne_type!r} must be a non-empty string")

    for field in GUIDELINE_FIELDS[2:]:
        if not isinstance(guideline[field], list):
            raise ValueError(f"Guideline field {field!r} for {ne_type!r} must be a list")
        if not all(isinstance(rule, str) and rule.strip() for rule in guideline[field]):
            raise ValueError(f"Guideline field {field!r} for {ne_type!r} must contain non-empty strings")

    return guideline


def load_guidelines(guidelines_path):
    with open(guidelines_path, "r", encoding="utf-8") as fp:
        guidelines = json.load(fp)

    if not isinstance(guidelines, dict):
        raise ValueError(f"Expected guideline JSON object at {guidelines_path}")

    return {ne_type: parse_guideline(ne_type, guideline) for ne_type, guideline in guidelines.items()}


def group_samples(samples):
    grouped = defaultdict(lambda: {"positive": [], "negative": []})
    for sample, output in samples:
        bucket = "positive" if output else "negative"
        grouped[sample["tagName"]][bucket].append(sample)
    return grouped


def format_rules(rules):
    if not rules:
        return "- None provided."
    return "\n".join(f"- {rule}" for rule in rules)


def build_instruction(ne_type, guideline):
    return f"""Extract all named entities of type {ne_type.upper()} from the text chunk you have read.

Use the hierarchical annotation guideline below to decide what should be extracted, what should be ignored, and how entity boundaries should be selected.

definition:
{guideline["definition"]}

include_rules:
{format_rules(guideline["include_rules"])}

exclude_rules:
{format_rules(guideline["exclude_rules"])}

boundary_rules:
{format_rules(guideline["boundary_rules"])}

confusable_type_rules:
{format_rules(guideline["confusable_type_rules"])}

Return a JSON list of entity strings for this entity type only. Return an empty list if no matching entities are present."""


def with_instruction(sample, guideline):
    ne_type = sample["tagName"]
    return {
        "doc_tag_pairID": sample["doc_tag_pairID"],
        "tagName": ne_type,
        "input": sample["input"],
        "instruction": build_instruction(ne_type, guideline),
        "output": sample["output"],
    }


def select_samples(candidates, count, rng, excluded_inputs=None):
    pool = list(candidates)
    rng.shuffle(pool)

    selected = []
    for sample in pool:
        if len(selected) >= count:
            break
        if excluded_inputs is not None and sample["input"] in excluded_inputs:
            continue
        selected.append(sample)

    return selected


def build_split_samples(grouped, guidelines, n_pos, n_neg, rng):
    train_samples = []
    dev_samples = []
    train_inputs = set()
    train_counts = {}
    insufficient = {}

    missing_guidelines = sorted(set(grouped) - set(guidelines))
    if missing_guidelines:
        raise ValueError(f"Missing guidelines for {len(missing_guidelines)} entity types: {missing_guidelines}")

    for ne_type in sorted(grouped):
        per_type = grouped[ne_type]
        guideline = guidelines[ne_type]

        selected_train_pos = select_samples(per_type["positive"], n_pos, rng)
        selected_train_neg = select_samples(per_type["negative"], n_neg, rng)

        for sample in selected_train_pos + selected_train_neg:
            train_inputs.add(sample["input"])
            train_samples.append(with_instruction(sample, guideline))

        train_counts[ne_type] = {
            "train_positive": len(selected_train_pos),
            "train_negative": len(selected_train_neg),
        }

    for ne_type in sorted(grouped):
        per_type = grouped[ne_type]
        guideline = guidelines[ne_type]

        selected_dev_pos = select_samples(per_type["positive"], n_pos, rng, excluded_inputs=train_inputs)
        selected_dev_neg = select_samples(per_type["negative"], n_neg, rng, excluded_inputs=train_inputs)

        for sample in selected_dev_pos + selected_dev_neg:
            dev_samples.append(with_instruction(sample, guideline))

        train_pos_count = train_counts[ne_type]["train_positive"]
        train_neg_count = train_counts[ne_type]["train_negative"]
        dev_pos_count = len(selected_dev_pos)
        dev_neg_count = len(selected_dev_neg)
        if train_pos_count < n_pos or train_neg_count < n_neg or dev_pos_count < n_pos or dev_neg_count < n_neg:
            insufficient[ne_type] = {
                "train_positive": train_pos_count,
                "train_negative": train_neg_count,
                "dev_positive": dev_pos_count,
                "dev_negative": dev_neg_count,
            }

    rng.shuffle(train_samples)
    rng.shuffle(dev_samples)
    return train_samples, dev_samples, insufficient


def ensure_output_paths(output_dir, overwrite):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    existing_paths = [path for path in [train_path, dev_path] if path.exists()]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Output already exists: {existing}. Pass --overwrite to replace it.")
    return train_path, dev_path


def write_jsonl(samples, output_path):
    with open(output_path, "w", encoding="utf-8", newline="\n") as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    guidelines_path = Path(args.guidelines_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines path does not exist: {guidelines_path}")
    if args.n_pos < 0 or args.n_neg < 0:
        raise ValueError("--n_pos and --n_neg must be non-negative")

    train_path, dev_path = ensure_output_paths(args.output_dir, args.overwrite)

    samples = load_samples(input_path)
    guidelines = load_guidelines(guidelines_path)
    grouped = group_samples(samples)

    rng = random.Random(args.seed)
    train_samples, dev_samples, insufficient = build_split_samples(
        grouped=grouped,
        guidelines=guidelines,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        rng=rng,
    )

    write_jsonl(train_samples, train_path)
    write_jsonl(dev_samples, dev_path)

    print(f"Read samples: {len(samples)}")
    print(f"Entity types: {len(grouped)}")
    print(f"Wrote train: {train_path} ({len(train_samples)} samples)")
    print(f"Wrote dev: {dev_path} ({len(dev_samples)} samples)")
    print(f"NE types with fewer than requested examples: {len(insufficient)}")
    if insufficient:
        print(json.dumps(insufficient, ensure_ascii=False, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
