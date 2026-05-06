"""
Sample short positive and full-context negative examples from PileNER 391 JSONL.

Positive examples prefer the sentence-selection heuristic in
data_handler_pileNER.py: select a short, clean sentence that contains at least
one gold span. If a type has too few strict short examples, the script falls
back to a containing sentence shorter than 200 characters, then to the original
input text. Negative examples keep the original input text unchanged.
"""

import argparse
import json
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_PATH = "data/pileNER/pileNER_391_all.jsonl"
DEFAULT_OUTPUT_PATH = "data/pileNER/pileNER_391_3pos_2neg_examples.json"

RELAXED_LENGTH_NE_TYPES = {
    "namespace",
    "import",
    "keyword",
    "surname",
    "file name",
    "header file",
    "related art",
    "boolean",
    "struct",
    "html attribute",
    "protein domain",
    "fieldterminology",
    "constant",
    "legal citation",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample per-NE short positives and full-context negatives from PileNER 391 JSONL."
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="Input PileNER 391 JSONL path.")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Output grouped JSON path.")
    parser.add_argument("--n_pos", type=int, default=3, help="Number of positive examples per NE type.")
    parser.add_argument("--n_neg", type=int, default=2, help="Number of negative examples per NE type.")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducible sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_path if it exists.")
    return parser.parse_args()


def split_into_sentences(passage):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?! \d)(?!\d)", passage)
    return [sentence for sentence in sentences if sentence.strip()]


def count_target_words(sentence, target_words):
    matches = re.finditer(r"\b(?:" + "|".join(map(re.escape, target_words)) + r")\b", sentence)
    target_words_found = [match.group() for match in matches]
    return len(target_words_found), target_words_found


def has_too_many_whitespaces(sentence, threshold=4):
    consecutive_whitespaces = re.findall(r"\s+", sentence)
    return any(len(whitespace) > threshold for whitespace in consecutive_whitespaces)


def has_too_many_newline(sentence, threshold=2):
    consecutive_newline = re.findall(r"\n+", sentence)
    return any(len(whitespace) >= threshold for whitespace in consecutive_newline)


def has_more_than_n_foreign_chars(sentence, threshold=2):
    foreign_char_count = sum(1 for char in sentence if ord(char) > 127)
    return foreign_char_count > threshold


def has_too_many_punctuations_and_digits(sentence, threshold=5):
    punctuation_count = sum(1 for char in sentence if char in string.punctuation or char.isdigit())
    return punctuation_count > threshold


def is_clean_positive_sentence(sentence):
    return (
        not has_too_many_whitespaces(sentence, 4)
        and not has_too_many_newline(sentence, 1)
        and not has_more_than_n_foreign_chars(sentence, 2)
        and not has_too_many_punctuations_and_digits(sentence, 10)
    )


def deduplicate_preserving_order(values):
    return list(dict.fromkeys(values))


def build_target_word_counts(sample, target_words):
    sentences = split_into_sentences(sample["input"])
    target_word_counts = []

    for sentence in sentences:
        occurrences_count, target_words_found = count_target_words(sentence, target_words)
        target_word_counts.append(
            {
                "sentence": sentence,
                "target_words_in_it": target_words_found,
                "occurrences_count": occurrences_count,
            }
        )

    return sorted(target_word_counts, key=lambda x: x["occurrences_count"], reverse=True)


def format_positive_example(sentence, target_words):
    return {
        "sentence": sentence,
        "entities": deduplicate_preserving_order(target_words),
    }


def get_strict_positive_example(sample, target_words):
    target_word_counts = build_target_word_counts(sample, target_words)
    ne_type = sample["tagName"]

    for candidate in target_word_counts:
        if candidate["occurrences_count"] == 0:
            continue

        sentence = candidate["sentence"]
        if 50 < len(sentence) < 100 and is_clean_positive_sentence(sentence):
            return format_positive_example(sentence, candidate["target_words_in_it"])

        if ne_type in RELAXED_LENGTH_NE_TYPES and len(sentence) < 200:
            return format_positive_example(sentence, candidate["target_words_in_it"])

    return None


def get_relaxed_positive_example(sample, target_words):
    target_word_counts = build_target_word_counts(sample, target_words)

    for candidate in target_word_counts:
        if candidate["occurrences_count"] == 0:
            continue

        sentence = candidate["sentence"]
        if len(sentence) < 200:
            return format_positive_example(sentence, candidate["target_words_in_it"])

    return None


def get_full_input_positive_example(sample, target_words):
    return format_positive_example(sample["input"], target_words)


def load_samples(input_path):
    samples = []
    with open(input_path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                output = json.loads(sample["output"])
            except (KeyError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid sample at {input_path}:{line_number}") from exc

            if not isinstance(output, list):
                raise ValueError(f"Expected output JSON list at {input_path}:{line_number}")

            samples.append((sample, output))

    return samples


def group_samples(samples):
    grouped = defaultdict(lambda: {"positive": [], "negative": []})
    for sample, output in samples:
        bucket = "positive" if output else "negative"
        grouped[sample["tagName"]][bucket].append((sample, output))
    return grouped


def sample_examples_for_type(ne_type, grouped_samples, n_pos, n_neg, rng):
    positive_pool = list(grouped_samples["positive"])
    negative_pool = list(grouped_samples["negative"])
    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)

    positive_examples = []
    used_positive_indexes = set()
    positive_strategies = [
        get_strict_positive_example,
        get_relaxed_positive_example,
        get_full_input_positive_example,
    ]

    for strategy in positive_strategies:
        if len(positive_examples) >= n_pos:
            break
        for sample_index, (sample, output) in enumerate(positive_pool):
            if len(positive_examples) >= n_pos:
                break
            if sample_index in used_positive_indexes:
                continue

            example = strategy(sample, output)
            if example is not None:
                positive_examples.append(example)
                used_positive_indexes.add(sample_index)

    negative_examples = []
    for sample, _ in negative_pool[:n_neg]:
        negative_examples.append(
            {
                "sentence": sample["input"],
                "entities": [],
            }
        )

    insufficient = {}
    if len(positive_examples) < n_pos:
        insufficient["positive_examples"] = len(positive_examples)
    if len(negative_examples) < n_neg:
        insufficient["negative_examples"] = len(negative_examples)

    return {
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
    }, insufficient


def build_examples(grouped, n_pos, n_neg, seed):
    rng = random.Random(seed)
    examples_by_type = {}
    insufficient_types = {}

    for ne_type in sorted(grouped):
        examples, insufficient = sample_examples_for_type(ne_type, grouped[ne_type], n_pos, n_neg, rng)
        examples_by_type[ne_type] = examples
        if insufficient:
            insufficient_types[ne_type] = insufficient

    return examples_by_type, insufficient_types


def write_json(examples_by_type, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(examples_by_type, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    samples = load_samples(input_path)
    grouped = group_samples(samples)
    examples_by_type, insufficient_types = build_examples(grouped, args.n_pos, args.n_neg, args.seed)
    write_json(examples_by_type, output_path)

    print(f"Read samples: {len(samples)}")
    print(f"Entity types: {len(examples_by_type)}")
    print(f"Wrote: {output_path}")
    print(f"NE types with fewer than requested examples: {len(insufficient_types)}")
    if insufficient_types:
        print(json.dumps(insufficient_types, ensure_ascii=False, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
