"""Tests for the ConceptPipelineDatasetPreparer class."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset_preparer import (
    ConceptPipelineDatasetPreparer,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId


def make_test_concept(concept_id: ConceptId, title: str = "Test Concept") -> Concept:
    """Create a test concept."""
    return Concept(
        id=concept_id,
        title=title,
        verbalization_check_guide="guide",
        removal_action="remove",
        addition_action="add",
    )


def make_test_dataset(
    concept_ids: list[ConceptId],
    input_ids: list[InputId],
    n_variations_per_concept_per_input: int = 1,
    existing_variations: dict | None = None,
) -> ConceptPipelineDataset:
    """Create a dataset with the specified configuration."""
    concepts = [
        make_test_concept(cid, f"Concept {i}") for i, cid in enumerate(concept_ids)
    ]
    return ConceptPipelineDataset(
        dataset_name="test",
        input_template="{loan_application}",
        input_parameters={},
        varying_input_param_name="loan_application",
        varying_inputs={iid: f"input {i}" for i, iid in enumerate(input_ids)},
        sanitized_varying_inputs={iid: f"input {i}" for i, iid in enumerate(input_ids)},
        concepts=concepts,
        deduplicated_concepts=concepts,
        n_variations_per_concept_per_input=n_variations_per_concept_per_input,
        removal_interpreted_as="negation",
        variations=existing_variations,
    )


class TrackingVariationsGenerator:
    """A mock variations generator that tracks requested pairs."""

    def __init__(self):
        self.requested_pairs: list[tuple[ConceptId, InputId]] = []
        self.config = {"mock": True}
        self.use_openai_batches = False

    async def generate_concept_variations(
        self,
        dataset: Any,
        variations_to_generate: dict[ConceptId, list[InputId]],
        removal_interpreted_as: str = "negation",
    ) -> dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]]:
        """Track all requested pairs and return mock variations."""
        result: dict[
            ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]
        ] = {}
        for concept_id, input_ids in variations_to_generate.items():
            for input_id in input_ids:
                self.requested_pairs.append((concept_id, input_id))
                if concept_id not in result:
                    result[concept_id] = {}
                if input_id not in result[concept_id]:
                    result[concept_id][input_id] = {}
                pair_id = VariationPairId()
                result[concept_id][input_id][pair_id] = VariationPair(
                    positive="positive", negative="negative"
                )
        return result


@pytest.mark.anyio
async def test_generate_variations_uses_dataset_config(tmp_path: Path):
    """Test that _generate_variations uses dataset.n_variations_per_concept_per_input,
    not self.n_variations_per_concept_per_input.

    This tests the fix for the bug where the preparer used its own instance value
    instead of the dataset's stored config value.
    """
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_id = InputId("00000000-0000-0000-0000-000000000002")

    # Dataset is configured for 2 variations per pair
    dataset = make_test_dataset(
        [concept_id],
        [input_id],
        n_variations_per_concept_per_input=2,
    )

    tracking_generator = TrackingVariationsGenerator()

    # Preparer is configured for 1 variation per pair (different from dataset!)
    preparer = ConceptPipelineDatasetPreparer(
        variations_generator=tracking_generator,  # type: ignore
        implicit_concept_detector=MagicMock(),
        input_clusterer=MagicMock(),
        output_dir=tmp_path,
        n_variations_per_concept_per_input=1,  # This should be IGNORED
    )

    # Monkey-patch save to avoid file operations
    dataset.save = MagicMock()

    await preparer._generate_variations(dataset)

    # Should request 2 variations (from dataset config), not 1 (from preparer)
    assert len(tracking_generator.requested_pairs) == 2
    assert all(
        pair == (concept_id, input_id) for pair in tracking_generator.requested_pairs
    )


@pytest.mark.anyio
async def test_generate_variations_respects_existing_count(tmp_path: Path):
    """Test that existing variations are counted and only missing ones are requested."""
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_id = InputId("00000000-0000-0000-0000-000000000002")
    existing_pair_id = VariationPairId("00000000-0000-0000-0000-000000000003")

    # Dataset already has 1 variation
    existing_variations = {
        concept_id: {
            input_id: {existing_pair_id: VariationPair(positive="p1", negative="n1")}
        }
    }

    # Dataset is configured for 2 variations per pair
    dataset = make_test_dataset(
        [concept_id],
        [input_id],
        n_variations_per_concept_per_input=2,
        existing_variations=existing_variations,
    )

    tracking_generator = TrackingVariationsGenerator()

    preparer = ConceptPipelineDatasetPreparer(
        variations_generator=tracking_generator,  # type: ignore
        implicit_concept_detector=MagicMock(),
        input_clusterer=MagicMock(),
        output_dir=tmp_path,
        n_variations_per_concept_per_input=2,
    )

    dataset.save = MagicMock()

    await preparer._generate_variations(dataset)

    # Should only request 1 more variation (2 target - 1 existing = 1 missing)
    assert len(tracking_generator.requested_pairs) == 1


@pytest.mark.anyio
async def test_generate_variations_skips_when_all_exist(tmp_path: Path):
    """Test that no variations are requested when all already exist."""
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_id = InputId("00000000-0000-0000-0000-000000000002")

    # Dataset already has 2 variations
    existing_variations = {
        concept_id: {
            input_id: {
                VariationPairId(): VariationPair(positive="p1", negative="n1"),
                VariationPairId(): VariationPair(positive="p2", negative="n2"),
            }
        }
    }

    # Dataset is configured for 2 variations per pair
    dataset = make_test_dataset(
        [concept_id],
        [input_id],
        n_variations_per_concept_per_input=2,
        existing_variations=existing_variations,
    )

    tracking_generator = TrackingVariationsGenerator()

    preparer = ConceptPipelineDatasetPreparer(
        variations_generator=tracking_generator,  # type: ignore
        implicit_concept_detector=MagicMock(),
        input_clusterer=MagicMock(),
        output_dir=tmp_path,
        n_variations_per_concept_per_input=2,
    )

    dataset.save = MagicMock()

    await preparer._generate_variations(dataset)

    # Should not request any variations
    assert len(tracking_generator.requested_pairs) == 0


@pytest.mark.anyio
async def test_generate_variations_handles_multiple_concepts_and_inputs(tmp_path: Path):
    """Test variation generation with multiple concepts and inputs."""
    concept_ids = [
        ConceptId("00000000-0000-0000-0000-000000000001"),
        ConceptId("00000000-0000-0000-0000-000000000002"),
    ]
    input_ids = [
        InputId("00000000-0000-0000-0000-00000000000a"),
        InputId("00000000-0000-0000-0000-00000000000b"),
    ]

    # Dataset is configured for 3 variations per pair
    dataset = make_test_dataset(
        concept_ids,
        input_ids,
        n_variations_per_concept_per_input=3,
    )

    tracking_generator = TrackingVariationsGenerator()

    preparer = ConceptPipelineDatasetPreparer(
        variations_generator=tracking_generator,  # type: ignore
        implicit_concept_detector=MagicMock(),
        input_clusterer=MagicMock(),
        output_dir=tmp_path,
        n_variations_per_concept_per_input=3,
    )

    dataset.save = MagicMock()

    await preparer._generate_variations(dataset)

    # Should request 3 variations for each of 4 (concept, input) pairs = 12 total
    assert len(tracking_generator.requested_pairs) == 12

    # Verify distribution: each (concept, input) pair should appear 3 times
    from collections import Counter

    pair_counts = Counter(tracking_generator.requested_pairs)
    for concept_id in concept_ids:
        for input_id in input_ids:
            assert pair_counts[(concept_id, input_id)] == 3
