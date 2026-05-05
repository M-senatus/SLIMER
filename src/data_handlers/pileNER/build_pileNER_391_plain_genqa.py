"""
Build a no-split PileNER 391-NE dataset from local conversation files.

This script intentionally stops before Definition & Guidelines generation. It
extracts PileNER QA pairs, applies the same NE normalization and zero-shot
filtering rules used by data_handler_pileNER.py, and writes plain GenQA-style
JSONL without an instruction field.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

DEFAULT_INPUT_DIR = "../dataset/Pile-NER-type"
DEFAULT_OUTPUT_PATH = "data/pileNER/pileNER_391_all.jsonl"
EXPECTED_NE_TYPES = 391


NE_TYPE_MAPPING = {
    "misc": None,
    "miscellaneous": None,
    "other": None,
    "unknown": None,
    "general": None,
    "entity type not specified": None,
    "entity type": None,
    "entity": None,
    "text": None,
    "import": None,
    "bacteria": "bacterium",
    "biological": "biological entity",
    "cell": "cell type",
    "cellular component": "cell component",
    "governmental body": "government body",
    "movie": "film",
    "work": "work of art",
    "musical group": "music group",
    "org": "organization",
    "anatomical_structure": "anatomical structure",
    "anatomicalstructure": "anatomical structure",
    "biological_process": "biological process",
    "body_part": "body part",
    "gpe": "geopolitical entity",
    "gene/protein": "gene",
    "work_of_art": "work of art",
    "job_title": "job title",
    "organisation": "organization",
    "chemical_substance": "chemical substance",
    "medical_condition": "medical condition",
    "medicalcondition": "medical condition",
    "fieldterminology": None,
    "cryptocurrency": "cryptocurrency",
    "demonym": "demonym",
    "norp": "norp",
}


ZERO_SHOT_REMOVED_TAGS = {
    "actor",
    "character",
    "genre",
    "song",
    "year",
    "dish",
    "restaurant",
    "algorithm",
    "field",
    "metric",
    "product",
    "programming language",
    "task",
    "university",
    "award",
    "book",
    "event",
    "magazine",
    "album",
    "band",
    "artist",
    "instrument",
    "music genre",
    "political party",
    "journal",
    "object",
    "chemical compound",
    "chemical",
    "element",
    "enzyme",
    "company",
    "legal",
}

# The historical helper lists both "instrument" and "musical instrument" as
# removable Music tags, but the checked-in top391NEs_definitions.json retains
# "musical instrument". Keep this script aligned with that audited 391-NE set.


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build no-split plain GenQA JSONL from local Pile-NER-type files."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Local Pile-NER-type directory.")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument(
        "--min_occurrences",
        type=int,
        default=100,
        help="Minimum positive answer span occurrences for an NE type before normalization.",
    )
    parser.add_argument(
        "--max_raw_samples",
        type=int,
        default=None,
        help="Optional limit on raw conversation samples for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_path if it exists.")
    return parser.parse_args()


def discover_data_files(input_dir):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(f"No supported data files ({suffixes}) found under {root}")
    return files


def iter_json_records(path):
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
        return

    with open(path, "r", encoding="utf-8") as fp:
        try:
            loaded = json.load(fp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file {path}: {exc}") from exc

    if isinstance(loaded, list):
        yield from loaded
    elif isinstance(loaded, dict) and "conversations" in loaded:
        yield loaded
    elif isinstance(loaded, dict):
        yielded = False
        for value in loaded.values():
            if isinstance(value, list):
                yield from value
                yielded = True
        if not yielded:
            raise ValueError(f"JSON object in {path} does not contain conversation samples")
    else:
        raise ValueError(f"Unsupported JSON top-level type in {path}: {type(loaded).__name__}")


def iter_parquet_records(paths):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading Parquet input requires the 'datasets' package. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset("parquet", data_files=[str(path) for path in paths], split="train")
    for sample in dataset:
        yield sample


def iter_raw_samples(files):
    json_files = [path for path in files if path.suffix.lower() in {".json", ".jsonl"}]
    parquet_files = [path for path in files if path.suffix.lower() == ".parquet"]

    for path in json_files:
        yield from iter_json_records(path)
    if parquet_files:
        yield from iter_parquet_records(parquet_files)


def find_start_positions(text, target_word):
    start_positions = []
    pattern = re.compile(r"\b" + re.escape(target_word) + r"\b")
    for match in pattern.finditer(text):
        start_positions.append({"answer_start": match.start(), "text": target_word})
    return start_positions


def extract_context_quests_answers(conversation):
    """
    Mirror data_handler_pileNER.extract_context_quests_answers without mutating input.
    """
    if not isinstance(conversation, list) or len(conversation) < 2:
        raise ValueError("Invalid conversation: expected a non-empty list")

    context_turn = conversation[0]
    if context_turn.get("from") == "human" and context_turn.get("value", "")[:5] == "Text:":
        context = context_turn["value"][len("Text: "):]
    else:
        raise ValueError("Invalid context or source in the conversation")

    if conversation[1].get("from") != "gpt":
        raise ValueError("Invalid conversation: expected GPT confirmation as the second turn")

    quests_answers = []
    for i in range(2, len(conversation), 2):
        if i + 1 >= len(conversation):
            raise ValueError("Invalid conversation: dangling human question without GPT answer")
        if conversation[i].get("from") != "human" or conversation[i + 1].get("from") != "gpt":
            raise ValueError("human-gpt non matched conversation")

        question = conversation[i]["value"]
        start_char_ne_type = len("What describes ")
        end_char_ne_type = question.find("in the text?") - 1
        if not question.startswith("What describes ") or end_char_ne_type < start_char_ne_type:
            raise ValueError(f"Invalid PileNER question format: {question}")

        ne_type = question[start_char_ne_type:end_char_ne_type].lower()
        normalized_question = "What describes " + ne_type + " in the text?"

        qa_answers = {"answer_start": [], "text": []}
        try:
            answers = ast.literal_eval(conversation[i + 1]["value"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid GPT answer list for question '{question}'") from exc
        if not isinstance(answers, list):
            raise ValueError(f"Invalid GPT answer type for question '{question}': expected list")

        for answer in answers:
            if not isinstance(answer, str):
                raise ValueError(f"Invalid answer value for question '{question}': expected string")
            for start_position in find_start_positions(context, answer):
                qa_answers["text"].append(start_position["text"])
                qa_answers["answer_start"].append(start_position["answer_start"])

        if len(qa_answers["text"]) != len(qa_answers["answer_start"]):
            raise ValueError("number of answer text not matching number of answer_start")

        quests_answers.append({"question": normalized_question, "ne_type": ne_type, "answers": qa_answers})

    return {"context": context, "questions_answers": quests_answers}


def validate_raw_sample(sample, sample_index):
    if not isinstance(sample, dict):
        raise ValueError(f"Raw sample #{sample_index} is not an object")
    if "conversations" not in sample:
        fields = ", ".join(sample.keys())
        raise ValueError(f"Raw sample #{sample_index} has no 'conversations' field. Fields: {fields}")


def build_qa_samples(files, max_raw_samples=None):
    qa_samples = []
    raw_count = 0
    for raw_index, raw_sample in enumerate(iter_raw_samples(files)):
        if max_raw_samples is not None and raw_count >= max_raw_samples:
            break
        validate_raw_sample(raw_sample, raw_index)
        raw_id = raw_sample.get("id", str(raw_index))
        parsed = extract_context_quests_answers(raw_sample["conversations"])
        for question_index, q_a in enumerate(parsed["questions_answers"]):
            qa_samples.append(
                {
                    "doc_question_pairID": f"{raw_id}:{question_index}",
                    "document_context": parsed["context"],
                    "tagName": q_a["ne_type"],
                    "question": q_a["question"],
                    "answers": q_a["answers"],
                }
            )
        raw_count += 1
    return qa_samples, raw_count


def get_frequent_ne_types(samples, min_occurrences):
    occurrence_counts = Counter()
    for sample in samples:
        occurrence_counts[sample["tagName"]] += len(sample["answers"]["text"])
    return {
        ne_type
        for ne_type, count in sorted(occurrence_counts.items(), key=lambda item: item[1], reverse=True)
        if count >= min_occurrences
    }


def normalize_and_filter_ne_types(samples, min_occurrences):
    frequent_ne_types = get_frequent_ne_types(samples, min_occurrences)
    normalized_samples = []

    for sample in samples:
        old_ne_type = sample["tagName"]
        ne_type = NE_TYPE_MAPPING.get(old_ne_type, old_ne_type)
        if ne_type is None or ne_type not in frequent_ne_types:
            continue
        if ne_type in ZERO_SHOT_REMOVED_TAGS:
            continue

        normalized_sample = dict(sample)
        normalized_sample["tagName"] = ne_type
        pattern = re.compile(re.escape(old_ne_type))
        normalized_sample["question"] = pattern.sub(ne_type.upper(), sample["question"])
        normalized_samples.append(normalized_sample)

    return normalized_samples, frequent_ne_types


def get_ordered_unique_answer_texts(sample):
    answers = sample["answers"]
    sorted_answers = sorted(zip(answers["answer_start"], answers["text"]), key=lambda item: item[0])
    answer_texts = [text for _, text in sorted_answers]
    return list(OrderedDict.fromkeys(answer_texts).keys())


def is_ascii_text(text):
    return isinstance(text, str) and text.isascii()


def is_ascii_sample(sample):
    answer_texts = get_ordered_unique_answer_texts(sample)
    return (
        is_ascii_text(sample["doc_question_pairID"])
        and is_ascii_text(sample["tagName"])
        and is_ascii_text(sample["document_context"])
        and all(is_ascii_text(answer) for answer in answer_texts)
    )


def filter_ascii_samples(samples):
    ascii_samples = []
    skipped = 0
    for sample in samples:
        if is_ascii_sample(sample):
            ascii_samples.append(sample)
        else:
            skipped += 1
    return ascii_samples, skipped


def format_genqa_without_instruction(sample):
    answer_texts = get_ordered_unique_answer_texts(sample)

    return {
        "doc_tag_pairID": sample["doc_question_pairID"],
        "tagName": sample["tagName"],
        "input": sample["document_context"],
        "output": json.dumps(answer_texts, ensure_ascii=False),
    }


def write_jsonl(samples, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(format_genqa_without_instruction(sample), ensure_ascii=True) + "\n")


def assert_expected_ne_count(samples):
    ne_types = sorted({sample["tagName"] for sample in samples})
    if len(ne_types) != EXPECTED_NE_TYPES:
        preview = ", ".join(ne_types[:20])
        raise ValueError(
            f"Expected {EXPECTED_NE_TYPES} NE types after filtering, got {len(ne_types)}. "
            f"First types: {preview}"
        )
    return ne_types


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    files = discover_data_files(args.input_dir)
    print(f"Found {len(files)} data files under {args.input_dir}")
    sys.stdout.flush()

    qa_samples, raw_count = build_qa_samples(files, max_raw_samples=args.max_raw_samples)
    print(f"Raw samples read: {raw_count}")
    print(f"Expanded QA samples: {len(qa_samples)}")
    sys.stdout.flush()

    filtered_samples, frequent_ne_types = normalize_and_filter_ne_types(qa_samples, args.min_occurrences)
    ascii_samples, skipped_non_ascii = filter_ascii_samples(filtered_samples)
    ne_types = assert_expected_ne_count(ascii_samples)
    print(f"NE types with >= {args.min_occurrences} positive spans before normalization: {len(frequent_ne_types)}")
    print(f"Final NE types: {len(ne_types)}")
    print(f"Skipped non-ASCII samples: {skipped_non_ascii}")
    print(f"Final samples: {len(ascii_samples)}")
    sys.stdout.flush()

    write_jsonl(ascii_samples, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
