"""Tests for the VariationsGenerator class."""

from typing import Any

import pytest

from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variations_generator import (
    VariationsGenerator,
)

# A simple generation prompt that only uses fields we provide
TEST_GENERATION_PROMPT = """Concept: {concept_description}
Addition: {addition_action}
{removal_label}: {removal_action}
Input: {test_input}"""


def make_test_dataset(
    concept_ids: list[ConceptId],
    input_ids: list[InputId],
) -> ConceptPipelineDataset:
    """Create a minimal dataset for testing variations generation."""
    concepts = [
        Concept(
            id=cid,
            title=f"Concept {i}",
            verbalization_check_guide="guide",
            removal_action="remove",
            addition_action="add",
        )
        for i, cid in enumerate(concept_ids)
    ]
    return ConceptPipelineDataset(
        dataset_name="test",
        input_template="{test_input}",
        input_parameters={},
        varying_input_param_name="test_input",
        varying_inputs={iid: f"input {i}" for i, iid in enumerate(input_ids)},
        sanitized_varying_inputs={iid: f"input {i}" for i, iid in enumerate(input_ids)},
        concepts=concepts,
        deduplicated_concepts=concepts,
    )


class MockVariationsGenerator(VariationsGenerator):
    """A mock generator that tracks calls and returns predictable responses."""

    def __init__(self, **kwargs: Any):
        # Use a simple generation prompt for testing
        kwargs.setdefault("generation_prompt", TEST_GENERATION_PROMPT)
        super().__init__(**kwargs)
        self.call_count = 0
        self.keys_received: list[tuple] = []

    async def _generate_batch_llm_response(
        self, key_to_messages: dict
    ) -> dict[tuple, str | None]:
        """Track keys and return valid response for each."""
        self.keys_received.extend(key_to_messages.keys())
        self.call_count += 1
        # Return a valid parseable response for each key
        return dict.fromkeys(
            key_to_messages,
            "POSITIVE_RESUME:\nPositive text\n\nNEGATIVE_RESUME:\nNegative text",
        )


@pytest.mark.anyio
async def test_multiple_variations_per_pair_generates_unique_keys():
    """Test that requesting multiple variations for the same (concept, input) pair
    generates unique keys and produces the expected number of variations.

    This tests the fix for the bug where key_to_messages[(concept_id, input_index)]
    would overwrite previous entries when the same pair was requested multiple times.
    """
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_id = InputId("00000000-0000-0000-0000-000000000002")

    dataset = make_test_dataset([concept_id], [input_id])

    generator = MockVariationsGenerator()

    # Request the same (concept, input) pair 3 times (simulating n_variations=3)
    variations_to_generate = {concept_id: [input_id, input_id, input_id]}

    result = await generator.generate_concept_variations(
        dataset, variations_to_generate
    )

    # Should have received 3 unique keys
    assert len(generator.keys_received) == 3

    # All keys should be unique (different variation_idx)
    assert len(set(generator.keys_received)) == 3

    # Keys should have the form (concept_id, input_id, variation_idx)
    for key in generator.keys_received:
        assert len(key) == 3
        assert key[0] == concept_id
        assert key[1] == input_id
        assert isinstance(key[2], int)

    # Variation indices should be 0, 1, 2
    variation_indices = sorted(key[2] for key in generator.keys_received)
    assert variation_indices == [0, 1, 2]

    # Result should contain 3 variation pairs for this (concept, input)
    assert concept_id in result
    assert input_id in result[concept_id]
    assert len(result[concept_id][input_id]) == 3


@pytest.mark.anyio
async def test_single_variation_per_pair():
    """Test that requesting a single variation works correctly."""
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_id = InputId("00000000-0000-0000-0000-000000000002")

    dataset = make_test_dataset([concept_id], [input_id])

    generator = MockVariationsGenerator()

    variations_to_generate = {concept_id: [input_id]}

    result = await generator.generate_concept_variations(
        dataset, variations_to_generate
    )

    # Should have received 1 key
    assert len(generator.keys_received) == 1

    # Key should have variation_idx = 0
    assert generator.keys_received[0] == (concept_id, input_id, 0)

    # Result should contain 1 variation pair
    assert len(result[concept_id][input_id]) == 1


@pytest.mark.anyio
async def test_multiple_concepts_multiple_inputs_multiple_variations():
    """Test complex scenario with multiple concepts, inputs, and variations per pair."""
    concept_ids = [
        ConceptId("00000000-0000-0000-0000-000000000001"),
        ConceptId("00000000-0000-0000-0000-000000000002"),
    ]
    input_ids = [
        InputId("00000000-0000-0000-0000-00000000000a"),
        InputId("00000000-0000-0000-0000-00000000000b"),
    ]

    dataset = make_test_dataset(concept_ids, input_ids)

    generator = MockVariationsGenerator()

    # Request 2 variations for each (concept, input) pair
    variations_to_generate = {
        concept_ids[0]: [input_ids[0], input_ids[0], input_ids[1], input_ids[1]],
        concept_ids[1]: [input_ids[0], input_ids[0], input_ids[1], input_ids[1]],
    }

    result = await generator.generate_concept_variations(
        dataset, variations_to_generate
    )

    # Should have 8 unique keys total (2 concepts * 2 inputs * 2 variations)
    assert len(generator.keys_received) == 8
    assert len(set(generator.keys_received)) == 8

    # Each (concept, input) pair should have 2 variations
    for concept_id in concept_ids:
        assert concept_id in result
        for input_id in input_ids:
            assert input_id in result[concept_id]
            assert len(result[concept_id][input_id]) == 2
