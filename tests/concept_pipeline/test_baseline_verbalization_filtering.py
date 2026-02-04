"""Tests verifying that concepts failing baseline verbalization are filtered out.

Concepts that are verbalized on baseline (i.e., fail the baseline verbalization check)
should NOT be processed in:
1. Variation bias testing
2. Variation verbalization analysis
3. Future stages
"""

from pathlib import Path

import pytest

import biases_in_the_blind_spot.concept_pipeline.variation_bias as vb
from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.statistics import check_futility_stopping
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId


def _make_concept(cid: ConceptId, title: str = "Test Concept") -> Concept:
    return Concept(
        id=cid,
        title=title,
        verbalization_check_guide="guide",
        removal_action="remove",
        addition_action="add",
    )


def _make_dataset_with_two_concepts(
    cid_unverbalized: ConceptId,
    cid_verbalized: ConceptId,
    iid: InputId,
    pid: VariationPairId,
) -> ConceptPipelineDataset:
    """Create a dataset with two concepts, both having variations."""
    concept_unverbalized = _make_concept(cid_unverbalized, "Unverbalized Concept")
    concept_verbalized = _make_concept(cid_verbalized, "Verbalized Concept")

    return ConceptPipelineDataset(
        dataset_name="test",
        input_template="{vary}",
        input_parameters={},
        varying_input_param_name="vary",
        varying_inputs={iid: "text"},
        sanitized_varying_inputs={iid: "text"},
        concepts=[concept_unverbalized, concept_verbalized],
        deduplicated_concepts=[concept_unverbalized, concept_verbalized],
        variations={
            cid_unverbalized: {iid: {pid: VariationPair("pos_uv", "neg_uv")}},
            cid_verbalized: {iid: {pid: VariationPair("pos_v", "neg_v")}},
        },
    )


class TrackingBiasTester(BiasTester):
    """A bias tester that tracks which concepts are processed."""

    def __init__(self):
        super().__init__(
            responses_generator=_StubResponsesGenerator(),  # type: ignore[arg-type]
            parse_response_fn=lambda _iid, resp: 1 if resp else 0,
        )
        self.concepts_processed: set[ConceptId] = set()

    def test_variatons_batch(self, **kwargs):
        variations_by_concept = kwargs.get("variations_by_concept", {})
        for cid in variations_by_concept.keys():
            self.concepts_processed.add(cid)
        return super().test_variatons_batch(**kwargs)


class _StubResponsesGenerator:
    model_name = "stub"

    def generate(
        self,
        input_template: str,
        input_parameters_list: list[dict[str, str]],
        **_,
    ):
        return ["response"] * len(input_parameters_list)


def test_variation_bias_skips_verbalized_concepts(tmp_path: Path):
    """Test that concepts verbalized on baseline are NOT processed in variation bias.

    This verifies the fix where test_variations_bias_for_stage now uses
    concept_ids_unverbalized_on_baseline instead of concepts_at_stage_start.
    """
    cid_unverbalized = ConceptId("00000000-0000-0000-0000-000000000001")
    cid_verbalized = ConceptId("00000000-0000-0000-0000-000000000002")
    iid = InputId("00000000-0000-0000-0000-00000000000a")
    pid = VariationPairId("00000000-0000-0000-0000-0000000000f1")

    dataset = _make_dataset_with_two_concepts(
        cid_unverbalized, cid_verbalized, iid, pid
    )

    # Stage has BOTH concepts at start, but only ONE is unverbalized on baseline
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid_unverbalized, cid_verbalized],
        # Only the unverbalized concept passed baseline verbalization check
        concept_ids_unverbalized_on_baseline=[cid_unverbalized],
        variation_bias_results=None,
        significant_concepts=None,
    )

    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
        significance_test="fisher",
    )

    tracking_tester = TrackingBiasTester()

    vb.test_variations_bias_for_stage(
        result,
        stage,
        dataset=dataset,
        bias_tester=tracking_tester,
        responses_generator=tracking_tester.responses_generator,
        output_dir=tmp_path,
    )

    # Only the unverbalized concept should have been processed
    assert cid_unverbalized in tracking_tester.concepts_processed
    assert cid_verbalized not in tracking_tester.concepts_processed

    # Only the unverbalized concept should have variation bias results
    assert stage.variation_bias_results is not None
    assert cid_unverbalized in stage.variation_bias_results
    assert cid_verbalized not in stage.variation_bias_results

    # Under the strict authority model, significant_concepts is set by futility stopping.
    stage.stage_significant_concepts_p_value_alpha = 0.05
    result.futility_stop_power_threshold = 0.10
    check_futility_stopping(dataset, result, stage)

    # Only the unverbalized concept should be in significant_concepts (after futility filtering)
    assert stage.significant_concepts is not None
    assert cid_unverbalized in stage.significant_concepts
    assert cid_verbalized not in stage.significant_concepts


def test_variation_bias_requires_unverbalized_baseline_set(tmp_path: Path):
    """Test that test_variations_bias_for_stage raises if unverbalized list is not set."""
    cid = ConceptId("00000000-0000-0000-0000-000000000001")
    iid = InputId("00000000-0000-0000-0000-00000000000a")
    pid = VariationPairId("00000000-0000-0000-0000-0000000000f1")

    concept = _make_concept(cid)
    dataset = ConceptPipelineDataset(
        dataset_name="test",
        input_template="{vary}",
        input_parameters={},
        varying_input_param_name="vary",
        varying_inputs={iid: "text"},
        sanitized_varying_inputs={iid: "text"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations={cid: {iid: {pid: VariationPair("pos", "neg")}}},
    )

    # Stage does NOT have concept_ids_unverbalized_on_baseline set
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=None,  # Not set!
        variation_bias_results=None,
    )

    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
        significance_test="fisher",
    )

    tracking_tester = TrackingBiasTester()

    # Should raise because concept_ids_unverbalized_on_baseline is None
    with pytest.raises(AssertionError, match="concept_ids_unverbalized_on_baseline"):
        vb.test_variations_bias_for_stage(
            result,
            stage,
            dataset=dataset,
            bias_tester=tracking_tester,
            responses_generator=tracking_tester.responses_generator,
            output_dir=tmp_path,
        )


def test_variation_bias_with_no_unverbalized_concepts(tmp_path: Path):
    """Test that variation bias handles the case when no concepts are unverbalized.

    If all concepts fail baseline verbalization, the unverbalized list is empty,
    and no concepts should be processed.
    """
    cid = ConceptId("00000000-0000-0000-0000-000000000001")
    iid = InputId("00000000-0000-0000-0000-00000000000a")
    pid = VariationPairId("00000000-0000-0000-0000-0000000000f1")

    concept = _make_concept(cid)
    dataset = ConceptPipelineDataset(
        dataset_name="test",
        input_template="{vary}",
        input_parameters={},
        varying_input_param_name="vary",
        varying_inputs={iid: "text"},
        sanitized_varying_inputs={iid: "text"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations={cid: {iid: {pid: VariationPair("pos", "neg")}}},
    )

    # Stage has concept at start, but it failed baseline verbalization (empty unverbalized list)
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[],  # Empty - all concepts were verbalized
        variation_bias_results=None,
    )

    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
        significance_test="fisher",
    )

    tracking_tester = TrackingBiasTester()

    vb.test_variations_bias_for_stage(
        result,
        stage,
        dataset=dataset,
        bias_tester=tracking_tester,
        responses_generator=tracking_tester.responses_generator,
        output_dir=tmp_path,
    )

    # No concepts should have been processed
    assert len(tracking_tester.concepts_processed) == 0

    # Variation bias results should be empty (but initialized)
    assert stage.variation_bias_results is not None
    assert len(stage.variation_bias_results) == 0
