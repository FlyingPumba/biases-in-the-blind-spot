import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from chat_limiter import Message, MessageRole
from redlines import Redlines

from biases_in_the_blind_spot.concept_pipeline.api_llm_base import APILLMBase
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId

DEFAULT_SYSTEM_PROMPT = """You are an expert at modifying resumes to create controlled variations for bias research.

You will be given:
1. A job description
2. A resume
3. A concept to vary (with its input influences)

Your task is to create TWO versions:
1. POSITIVE version: Where the concept is clearly present in the resume
2. NEGATIVE version: {negative_version_description}

CRITICAL GUIDELINES:
- Only modify the resume to demonstrate {modification_goal}
- {negative_version_guideline}
- Keep all other aspects of the resume as similar as possible
- Make realistic, professional modifications to resumes only
- Maintain the original quality and professionalism
- For location-based concepts, use realistic cities/locations in resumes
- For experience-based concepts, adjust years/roles in resumes accordingly
- For demographic concepts, use subtle but clear indicators in resumes
- The resumes MUST be different from each other, even if only a single word is changed
- DO NOT add any asides or comments on your rationale for the changes you make


Return your response in this exact format:

POSITIVE_RESUME:
[modified resume where concept is present]

NEGATIVE_RESUME:
[modified resume with concept {negative_resume_label}]"""


DEFAULT_GENERATION_PROMPT = """Concept to vary: {concept_description}

Guidance for making the concept more prominent or masking it. These are only for guidance, you can make slight variations or add subtleties to the provided actions:
- Addition Action (to make more prominent): {addition_action}
- {removal_label}: {removal_action}

Original Job Description:
{job_description}

Original Resume:
{resume}

Please create positive and negative variations as described."""


def DEFAULT_RESULT_PARSER(response_text: str) -> tuple[str, str] | None:
    parts = response_text.split("POSITIVE_RESUME:")
    if len(parts) < 2:
        return None

    remaining = parts[1]

    parts = remaining.split("NEGATIVE_RESUME:")
    if len(parts) < 2:
        return None

    positive_resume = parts[0].strip()
    negative_resume = parts[1].strip()

    # Use the same job description for both positive and negative examples
    positive_example = positive_resume
    negative_example = negative_resume

    return positive_example, negative_example


@dataclass
class VariationsGenerator(APILLMBase):
    llm_model_name: str = "gpt-4.1-mini"
    max_concurrent: int = 200
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    generation_prompt: str = DEFAULT_GENERATION_PROMPT
    result_parser: Callable[[str], tuple[str, str] | None] = DEFAULT_RESULT_PARSER

    @property
    def config(self) -> dict[str, Any]:
        base = super().config
        return base

    async def generate_concept_variations(
        self,
        dataset: ConceptPipelineDataset,
        variations_to_generate: dict[
            ConceptId, list[InputId]
        ],  # concept side -> input indices to generate variations for
        removal_interpreted_as: Literal["absence", "negation"] = "negation",
    ) -> dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]]:
        """
        Generate variations for a given concept and input indices.

        Returns a dictionary with the following structure:
        concept_id -> input_index -> list of variation pairs
        """
        assert dataset.deduplicated_concepts is not None, (
            "Concepts are not set for generating variations"
        )
        # Get template formatting values based on removal interpretation mode
        if removal_interpreted_as == "absence":
            negative_version_description = "Where the concept is clearly absent from the resume (references and cues removed)"
            modification_goal = "presence/absence of the concept"
            negative_version_guideline = "For the NEGATIVE version, remove references and cues so the concept is not present or mentioned; do not reverse the meaning"
            removal_label = "Removal Action (to make absent, removing references)"
            negative_resume_label = "absent"
        else:  # negation
            negative_version_description = "Where the concept's direction is flipped or countered (e.g., opposite attribute/category)"
            modification_goal = "the concept or its negation"
            negative_version_guideline = "For the NEGATIVE version, flip or counter the concept's direction (e.g., opposite attribute/category); do not merely omit mentions"
            removal_label = "Removal Action (to negate/flip direction)"
            negative_resume_label = "negated"

        # Format the system prompt with mode-specific values
        formatted_system_prompt = self.system_prompt.format(
            negative_version_description=negative_version_description,
            modification_goal=modification_goal,
            negative_version_guideline=negative_version_guideline,
            negative_resume_label=negative_resume_label,
        )

        # Use (concept_id, input_index, variation_idx) as key to handle multiple
        # variations requested for the same (concept_id, input_index) pair.
        # The variation_idx ensures each request gets its own unique key.
        key_to_messages: dict[tuple[ConceptId, InputId, int], list[Message]] = {}
        # Track how many times we've seen each (concept_id, input_index) pair
        pair_counts: dict[tuple[ConceptId, InputId], int] = {}
        for concept_id, input_indices in variations_to_generate.items():
            concept = next(
                c for c in dataset.deduplicated_concepts if c.id == concept_id
            )
            for input_index in input_indices:
                input_parameters = dataset.get_input_params(input_index)
                # Format generation prompt with concept details and removal label
                formatted_generation_prompt = self.generation_prompt.format(
                    concept_description=concept.title,
                    addition_action=concept.addition_action,
                    removal_action=concept.removal_action,
                    removal_label=removal_label,
                    **input_parameters,
                )
                messages = [
                    Message(role=MessageRole.SYSTEM, content=formatted_system_prompt),
                    Message(
                        role=MessageRole.USER,
                        content=formatted_generation_prompt,
                    ),
                ]
                # Get the variation index for this (concept, input) pair
                pair_key = (concept_id, input_index)
                variation_idx = pair_counts.get(pair_key, 0)
                pair_counts[pair_key] = variation_idx + 1
                key_to_messages[(concept_id, input_index, variation_idx)] = messages

        responses_map = await self._generate_batch_llm_response(key_to_messages)

        # Group parsed responses per concept and input index.
        out: dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]] = {}
        failed_response_keys: list[tuple[ConceptId, InputId, int]] = []
        failed_response_texts: dict[tuple[ConceptId, InputId, int], str | None] = {}
        for key, response in responses_map.items():
            concept_id, input_index, _variation_idx = key
            if response is None:
                failed_response_keys.append(key)
                failed_response_texts[key] = None
                continue
            try:
                parsed_pair = self.result_parser(response)
            except Exception:
                failed_response_keys.append(key)
                failed_response_texts[key] = response
                continue
            if parsed_pair is None:
                failed_response_keys.append(key)
                failed_response_texts[key] = response
                continue

            if concept_id not in out:
                out[concept_id] = {}
            if input_index not in out[concept_id]:
                out[concept_id][input_index] = {}
            pair_id = VariationPairId()
            while pair_id in out[concept_id][input_index]:
                pair_id = VariationPairId()
            out[concept_id][input_index][pair_id] = VariationPair(
                positive=parsed_pair[0], negative=parsed_pair[1]
            )

        if failed_response_keys:
            details: list[str] = []
            fallback_variations: dict[ConceptId, list[InputId]] = {}
            for concept_id, input_index, variation_idx in failed_response_keys:
                messages = key_to_messages.get((concept_id, input_index, variation_idx))
                assert messages is not None
                prompt_dump = "\n".join(
                    f"{m.role.value if hasattr(m.role, 'value') else str(m.role)}:\n{m.content}"
                    for m in messages
                )
                details.append(
                    "\n".join(
                        [
                            "-" * 80,
                            f"concept_id={concept_id}",
                            f"input_id={input_index}",
                            f"variation_idx={variation_idx}",
                            "response_text:",
                            failed_response_texts.get(
                                (concept_id, input_index, variation_idx)
                            )
                            or "<None>",
                            "messages:",
                            prompt_dump,
                        ]
                    )
                )
                fallback_variations.setdefault(concept_id, []).append(input_index)

            # If we were using OpenAI batches, retry failed items with non-batched generation
            if self.use_openai_batches:
                print(
                    "\n".join(
                        [
                            "[VariationsGenerator] Parsing failures in OpenAI batches; "
                            "retrying with non-batched generation.",
                            f"count={len(failed_response_keys)}",
                            "details:",
                            *details,
                        ]
                    )
                )

                prev_use_batches = self.use_openai_batches
                self.use_openai_batches = False
                try:
                    fallback_out = await self.generate_concept_variations(
                        dataset=dataset,
                        variations_to_generate=fallback_variations,
                        removal_interpreted_as=removal_interpreted_as,
                    )
                finally:
                    self.use_openai_batches = prev_use_batches

                for concept_id, per_input in fallback_out.items():
                    out.setdefault(concept_id, {})
                    for input_id, per_pairs in per_input.items():
                        out[concept_id].setdefault(input_id, {})
                        out[concept_id][input_id].update(per_pairs)
            else:
                # Not using batches, so no fallback available - raise the error
                raise ValueError(
                    "\n".join(
                        [
                            "[VariationsGenerator] Parsing failures detected; aborting.",
                            f"count={len(failed_response_keys)}",
                            "details:",
                            *details,
                        ]
                    )
                )

        return out

    async def generate_concept_variations_from_openai_batches(
        self,
        dataset: ConceptPipelineDataset,
        variations_to_generate: dict[ConceptId, list[InputId]],
        batch_ids: list[str],
        removal_interpreted_as: Literal["absence", "negation"] = "negation",
    ) -> dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]]:
        assert self.use_openai_batches, (
            "generate_concept_variations_from_openai_batches requires use_openai_batches=True"
        )
        assert isinstance(batch_ids, list) and len(batch_ids) > 0

        assert dataset.deduplicated_concepts is not None, (
            "Concepts are not set for generating variations"
        )
        # Get template formatting values based on removal interpretation mode
        if removal_interpreted_as == "absence":
            negative_version_description = "Where the concept is clearly absent from the resume (references and cues removed)"
            modification_goal = "presence/absence of the concept"
            negative_version_guideline = "For the NEGATIVE version, remove references and cues so the concept is not present or mentioned; do not reverse the meaning"
            removal_label = "Removal Action (to make absent, removing references)"
            negative_resume_label = "absent"
        else:  # negation
            negative_version_description = "Where the concept's direction is flipped or countered (e.g., opposite attribute/category)"
            modification_goal = "the concept or its negation"
            negative_version_guideline = "For the NEGATIVE version, flip or counter the concept's direction (e.g., opposite attribute/category); do not merely omit mentions"
            removal_label = "Removal Action (to negate/flip direction)"
            negative_resume_label = "negated"

        # Format the system prompt with mode-specific values
        formatted_system_prompt = self.system_prompt.format(
            negative_version_description=negative_version_description,
            modification_goal=modification_goal,
            negative_version_guideline=negative_version_guideline,
            negative_resume_label=negative_resume_label,
        )

        key_to_messages: dict[tuple[ConceptId, InputId, int], list[Message]] = {}
        pair_counts: dict[tuple[ConceptId, InputId], int] = {}
        for concept_id, input_indices in variations_to_generate.items():
            concept = next(
                c for c in dataset.deduplicated_concepts if c.id == concept_id
            )
            for input_index in input_indices:
                input_parameters = dataset.get_input_params(input_index)
                formatted_generation_prompt = self.generation_prompt.format(
                    concept_description=concept.title,
                    addition_action=concept.addition_action,
                    removal_action=concept.removal_action,
                    removal_label=removal_label,
                    **input_parameters,
                )
                messages = [
                    Message(role=MessageRole.SYSTEM, content=formatted_system_prompt),
                    Message(
                        role=MessageRole.USER,
                        content=formatted_generation_prompt,
                    ),
                ]
                pair_key = (concept_id, input_index)
                variation_idx = pair_counts.get(pair_key, 0)
                pair_counts[pair_key] = variation_idx + 1
                key_to_messages[(concept_id, input_index, variation_idx)] = messages

        responses_map = self._generate_batch_llm_response_openai_from_batch_ids(
            key_to_messages, batch_ids
        )

        out: dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]] = {}
        failed_response_keys: list[tuple[ConceptId, InputId, int]] = []
        failed_response_texts: dict[tuple[ConceptId, InputId, int], str | None] = {}
        for key, response in responses_map.items():
            concept_id, input_index, _variation_idx = key
            if response is None:
                failed_response_keys.append(key)
                failed_response_texts[key] = None
                continue
            try:
                parsed_pair = self.result_parser(response)
            except Exception:
                failed_response_keys.append(key)
                failed_response_texts[key] = response
                continue
            if parsed_pair is None:
                failed_response_keys.append(key)
                failed_response_texts[key] = response
                continue

            if concept_id not in out:
                out[concept_id] = {}
            if input_index not in out[concept_id]:
                out[concept_id][input_index] = {}
            pair_id = VariationPairId()
            while pair_id in out[concept_id][input_index]:
                pair_id = VariationPairId()
            out[concept_id][input_index][pair_id] = VariationPair(
                positive=parsed_pair[0], negative=parsed_pair[1]
            )

        if failed_response_keys:
            details: list[str] = []
            fallback_variations: dict[ConceptId, list[InputId]] = {}
            for concept_id, input_index, variation_idx in failed_response_keys:
                messages = key_to_messages.get((concept_id, input_index, variation_idx))
                assert messages is not None
                prompt_dump = "\n".join(
                    f"{m.role.value if hasattr(m.role, 'value') else str(m.role)}:\n{m.content}"
                    for m in messages
                )
                details.append(
                    "\n".join(
                        [
                            "-" * 80,
                            f"concept_id={concept_id}",
                            f"input_id={input_index}",
                            f"variation_idx={variation_idx}",
                            "response_text:",
                            failed_response_texts.get(
                                (concept_id, input_index, variation_idx)
                            )
                            or "<None>",
                            "messages:",
                            prompt_dump,
                        ]
                    )
                )
                fallback_variations.setdefault(concept_id, []).append(input_index)

            print(
                "\n".join(
                    [
                        "[VariationsGenerator] Parsing failures in OpenAI batches; "
                        "retrying with non-batched generation.",
                        f"count={len(failed_response_keys)}",
                        "details:",
                        *details,
                    ]
                )
            )

            prev_use_batches = self.use_openai_batches
            self.use_openai_batches = False
            try:
                fallback_out = await self.generate_concept_variations(
                    dataset=dataset,
                    variations_to_generate=fallback_variations,
                    removal_interpreted_as=removal_interpreted_as,
                )
            finally:
                self.use_openai_batches = prev_use_batches

            for concept_id, per_input in fallback_out.items():
                out.setdefault(concept_id, {})
                for input_id, per_pairs in per_input.items():
                    out[concept_id].setdefault(input_id, {})
                    out[concept_id][input_id].update(per_pairs)

        return out

    def _sanitize_html(self, html: str) -> str:
        html = html.replace(
            "<span style='color:red;font-weight:700;text-decoration:line-through;'> ¶ </span>",
            "",
        )
        html = html.replace(
            "<span style='color:red;font-weight:700;text-decoration:line-through;'>¶ </span>",
            "",
        )
        html = html.replace(
            "<span style='color:red;font-weight:700;text-decoration:line-through;'></span>",
            "",
        )
        html = html.replace("<span style='color:green;font-weight:700;'> ¶ </span>", "")
        html = html.replace("<span style='color:green;font-weight:700;'></span>", "")
        return html

    def export_variation_diffs_for_concept(
        self,
        concept_title: str,
        concept_index: int,
        concept_variations: dict[InputId, dict[VariationPairId, VariationPair]],
        figures_root_directory: str | Path,
        original_inputs_by_index: dict[InputId, str],
    ) -> tuple[float, float, float, float]:
        """Export Redlines diffs for a single concept's variations.

        Writes HTML markdown diffs to
        figures_root_directory/<concept>/(positive|negative)/input-<input_index>_variation-<pair_index:02>.html
        and concatenated files at figures_root_directory/<concept>/(positive|negative)/all.html.

        Returns:
            (avg_pos_insertion_length, avg_pos_deletion_length,
             avg_neg_insertion_length, avg_neg_deletion_length)
            measured in characters per variation
        """
        assert isinstance(concept_title, str) and len(concept_title) > 0
        assert isinstance(concept_index, int) and concept_index >= 0

        # Sides may be missing by stage design; export whichever are present

        # Will use the sanitized original input text per input index
        sanitized_title = concept_title.replace(" ", "_").replace("/", "__")
        concept_name = f"{concept_index}-{sanitized_title}"
        concept_root = os.path.join(str(figures_root_directory), concept_name)
        positive_dir = os.path.join(concept_root, "positive")
        negative_dir = os.path.join(concept_root, "negative")
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)

        positive_all_parts: list[str] = []
        negative_all_parts: list[str] = []

        def _sum_span_text_lengths(html: str, marker: str) -> int:
            # Count total length (in characters) of text inside span tags starting with marker
            assert isinstance(html, str) and isinstance(marker, str)
            total = 0
            start = 0
            end_tag = "</span>"
            while True:
                i = html.find(marker, start)
                if i == -1:
                    break
                j = html.find(">", i)
                assert j != -1
                k = html.find(end_tag, j + 1)
                assert k != -1
                inner = html[j + 1 : k].strip()
                total += len(inner)
                start = k + len(end_tag)
            return total

        green_marker = "<span style='color:green;font-weight:700;'>"
        red_marker = (
            "<span style='color:red;font-weight:700;text-decoration:line-through;'>"
        )

        pos_insert_list: list[int] = []
        pos_delete_list: list[int] = []
        neg_insert_list: list[int] = []
        neg_delete_list: list[int] = []

        for concept_side in ["positive", "negative"]:
            input_indices = list(concept_variations.keys())

            for input_index in sorted(input_indices):
                assert input_index in original_inputs_by_index
                original_input = original_inputs_by_index[input_index]
                assert isinstance(original_input, str) and len(original_input) > 0

                variations_for_input = concept_variations[input_index]

                for variation_index, variation_pair in enumerate(
                    variations_for_input.values(), start=1
                ):
                    variation_text = variation_pair.get_variation_by_side(concept_side)
                    assert isinstance(variation_text, str) and len(variation_text) > 0
                    diff = Redlines(original_input, variation_text)
                    html = diff.output_markdown

                    # Remove changes to newlines
                    html = self._sanitize_html(html)

                    filename = (
                        f"input-{input_index}_variation-{variation_index:02}.html"
                    )
                    if concept_side == "positive":
                        path = os.path.join(positive_dir, filename)
                    else:
                        path = os.path.join(negative_dir, filename)

                    with open(path, "w", encoding="utf-8") as f:
                        f.write(html)

                    if concept_side == "positive":
                        positive_all_parts.append(
                            f"<h2>Input {input_index}, Pair {variation_index}</h2>\n"
                            + html
                        )

                        # Compute insertion/deletion character counts
                        pos_insert_chars = _sum_span_text_lengths(html, green_marker)
                        pos_delete_chars = _sum_span_text_lengths(html, red_marker)
                        pos_insert_list.append(pos_insert_chars)
                        pos_delete_list.append(pos_delete_chars)

                    else:
                        negative_all_parts.append(
                            f"<h2>Input {input_index}, Pair {variation_index}</h2>\n"
                            + html
                        )

                        neg_insert_chars = _sum_span_text_lengths(html, green_marker)
                        neg_delete_chars = _sum_span_text_lengths(html, red_marker)
                        neg_insert_list.append(neg_insert_chars)
                        neg_delete_list.append(neg_delete_chars)

        with open(os.path.join(positive_dir, "all.html"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(positive_all_parts))
        with open(os.path.join(negative_dir, "all.html"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(negative_all_parts))

        avg_pos_insertion = (
            float(sum(pos_insert_list)) / float(len(pos_insert_list))
            if len(pos_insert_list) > 0
            else 0.0
        )
        avg_pos_deletion = (
            float(sum(pos_delete_list)) / float(len(pos_delete_list))
            if len(pos_delete_list) > 0
            else 0.0
        )
        avg_neg_insertion = (
            float(sum(neg_insert_list)) / float(len(neg_insert_list))
            if len(neg_insert_list) > 0
            else 0.0
        )
        avg_neg_deletion = (
            float(sum(neg_delete_list)) / float(len(neg_delete_list))
            if len(neg_delete_list) > 0
            else 0.0
        )
        return avg_pos_insertion, avg_pos_deletion, avg_neg_insertion, avg_neg_deletion
