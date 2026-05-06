"""Prompt helpers for hierarchical NER annotation guidelines.

The generated guideline follows:
G_t = (t, d_t, I_t, O_t, B_t, C_t)
where the JSON fields are entity_type, definition, include_rules,
exclude_rules, boundary_rules, and confusable_type_rules.
"""

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


SYSTEM_MESSAGE = (
    "You are a careful NER annotation guideline designer. You generate precise hierarchical annotation guidelines for named entity recognition. Output only valid JSON. Do not include markdown, comments, or explanations outside the JSON object."
)

GUIDELINE_FIELDS = [
    "entity_type",
    "definition",
    "include_rules",
    "exclude_rules",
    "boundary_rules",
    "confusable_type_rules",
]

LOCATION_EXAMPLE_PROMPT = """We are constructing hierarchical annotation guidelines for a named entity type.

Your output must use these JSON fields:
- entity_type: the entity type name.
- definition: the semantic definition of the entity type.
- include_rules: rules describing what should be annotated.
- exclude_rules: rules describing what should not be annotated.
- boundary_rules: rules describing span start/end decisions.
- confusable_type_rules: rules distinguishing this type from confusable entity types.

Named Entity:
location

Positive examples:
[
  {
    "sentence": "He was awarded honorary degree from the University of Cambridge in Cambridge, UK.",
    "entities": ["Cambridge", "UK"]
  },
  {
    "sentence": "The Big Bend Country is part of the larger Columbia Country.",
    "entities": ["Big Bend Country", "Columbia Country"]
  }
]

Negative or boundary examples:
[
  {
    "sentence": "You should not label here or there as location entities.",
    "entities": []
  }
]

Instructions:
1. Write a concise but specific definition for the entity type.
2. Derive include rules from the positive examples and the entity type semantics.
3. Derive exclude rules for related, generic, overlapping, or misleading expressions that should not be annotated.
4. Write boundary rules that specify exactly what words should be included or excluded from the entity span.
5. Write confusable type rules that distinguish the current type from likely confusable entity types by semantic object, context role, and annotation scope, using only the entity type name and examples as evidence.
6. Make the rules practical for NER annotation and useful for an LLM performing single-type extraction.
7. Avoid vague rules such as "annotate relevant entities" unless they are made concrete.
8. Do not invent dataset-specific labels beyond the given entity type.
9. Return only a JSON object following the schema below.

Output JSON schema:
{
  "entity_type": "",
  "definition": "",
  "include_rules": [],
  "exclude_rules": [],
  "boundary_rules": [],
  "confusable_type_rules": []
}"""

LOCATION_EXAMPLE_ANSWER = {
    "entity_type": "location",
    "definition": (
        "A location is a geographic place, region, landmark, or geopolitical "
        "area that denotes a concrete place in the physical world."
    ),
    "include_rules": [
        "Annotate names of countries, cities, regions, territories, geographic areas, and named landmarks.",
        "Annotate full multi-word geographic names when the complete phrase functions as the place name.",
    ],
    "exclude_rules": [
        "Do not annotate vague spatial expressions such as here, there, nearby, or abroad.",
        "Do not annotate organization names merely because they contain a location word.",
        "Do not annotate person names that contain words that can also be locations.",
    ],
    "boundary_rules": [
        "Include all tokens that are part of the official or conventional place name.",
        "Exclude surrounding prepositions, determiners, and descriptive context unless they are part of the name.",
        "For nested place mentions, annotate the span according to the dataset convention shown in examples.",
    ],
    "confusable_type_rules": [
        "Distinguish locations from organizations: annotate Cambridge as a location when it refers to the city, but not when it refers to the university as an institution.",
        "Distinguish locations from persons: names such as Jordan or Paris should be annotated as locations only when the context refers to the place.",
    ],
}


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def normalize_positive_examples(examples: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return examples in the notebook's prompt-ready sentence/entities shape."""
    normalized = []
    for example in examples:
        if "entities" in example:
            entities = example["entities"]
        else:
            entities = example.get("target_words_in_it", [])
        normalized.append(
            {
                "sentence": example.get("sentence", ""),
                "entities": list(entities),
            }
        )
    return normalized


def examples_from_exemplary_data(exemplary_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract positive examples from sentences_per_ne_type entries."""
    return normalize_positive_examples(exemplary_data.get("sentences", []))


def build_hierarchical_guideline_prompt(
    entity_type: str,
    positive_examples: Sequence[Mapping[str, Any]],
    negative_or_boundary_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    confusable_entity_types: Optional[Sequence[str]] = None,
    hints: str = "",
) -> str:
    """Build the target user prompt for one entity type."""
    positive_examples_json = _json_block(normalize_positive_examples(positive_examples))
    negative_examples_json = _json_block(negative_or_boundary_examples or [])
    optional_context = []
    if confusable_entity_types:
        optional_context.append(
            "Confusable entity types:\n" + _json_block(confusable_entity_types)
        )
    if hints:
        optional_context.append("Additional hints:\n" + hints)
    optional_context_text = "\n\n" + "\n\n".join(optional_context) if optional_context else ""

    return f"""We are constructing hierarchical annotation guidelines for a named entity type.

Your output must use these JSON fields:
- entity_type: the entity type name.
- definition: the semantic definition of the entity type.
- include_rules: rules describing what should be annotated.
- exclude_rules: rules describing what should not be annotated.
- boundary_rules: rules describing span start/end decisions.
- confusable_type_rules: rules distinguishing this type from confusable entity types.

Named Entity:
{entity_type}

Positive examples:
{positive_examples_json}

Negative or boundary examples:
{negative_examples_json}{optional_context_text}

Instructions:
1. Write a concise but specific definition for the entity type.
2. Derive include rules from the positive examples and the entity type semantics.
3. Derive exclude rules for related, generic, overlapping, or misleading expressions that should not be annotated.
4. Write boundary rules that specify exactly what words should be included or excluded from the entity span.
5. Write confusable type rules that distinguish the current type from likely confusable entity types by semantic object, context role, and annotation scope, using only the entity type name and examples as evidence.
6. Make the rules practical for NER annotation and useful for an LLM performing single-type extraction.
7. Avoid vague rules such as "annotate relevant entities" unless they are made concrete.
8. Do not invent dataset-specific labels beyond the given entity type.
9. Return only a JSON object following the schema below.

Output JSON schema:
{{
  "entity_type": "",
  "definition": "",
  "include_rules": [],
  "exclude_rules": [],
  "boundary_rules": [],
  "confusable_type_rules": []
}}"""


def build_hierarchical_guideline_messages(
    entity_type: str,
    positive_examples: Sequence[Mapping[str, Any]],
    negative_or_boundary_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    confusable_entity_types: Optional[Sequence[str]] = None,
    hints: str = "",
    include_few_shot: bool = True,
) -> List[Dict[str, str]]:
    """Build chat-completion messages for hierarchical guideline generation."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if include_few_shot:
        messages.extend(
            [
                {"role": "user", "content": LOCATION_EXAMPLE_PROMPT},
                {
                    "role": "assistant",
                    "content": _json_block(LOCATION_EXAMPLE_ANSWER),
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": build_hierarchical_guideline_prompt(
                entity_type=entity_type,
                positive_examples=positive_examples,
                negative_or_boundary_examples=negative_or_boundary_examples,
                confusable_entity_types=confusable_entity_types,
                hints=hints,
            ),
        }
    )
    return messages


def parse_guideline_json(guideline: Union[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Parse a guideline JSON string or dict and validate the expected schema."""
    if isinstance(guideline, str):
        guideline = json.loads(guideline)
    guideline = dict(guideline)

    missing_fields = [field for field in GUIDELINE_FIELDS if field not in guideline]
    if missing_fields:
        raise ValueError(f"Missing guideline fields: {missing_fields}")

    if not isinstance(guideline["entity_type"], str) or not guideline["entity_type"].strip():
        raise ValueError("entity_type must be a non-empty string")
    if not isinstance(guideline["definition"], str) or not guideline["definition"].strip():
        raise ValueError("definition must be a non-empty string")

    for field in GUIDELINE_FIELDS[2:]:
        if not isinstance(guideline[field], list):
            raise ValueError(f"{field} must be a list")
        if not all(isinstance(rule, str) and rule.strip() for rule in guideline[field]):
            raise ValueError(f"{field} must contain only non-empty strings")

    return guideline


def flatten_guideline_for_slimer(guideline: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    """Map a hierarchical guideline back to SLIMER's flat Definition/Guidelines."""
    parsed = parse_guideline_json(guideline)
    rule_parts = []
    for title, field in [
        ("Include", "include_rules"),
        ("Exclude", "exclude_rules"),
        ("Boundary", "boundary_rules"),
        ("Confusable types", "confusable_type_rules"),
    ]:
        if parsed[field]:
            rule_parts.append(f"{title}: " + " ".join(parsed[field]))
    return {
        "Definition": parsed["definition"],
        "Guidelines": " ".join(rule_parts),
    }
