from pathlib import Path

import pytest

from biases_in_the_blind_spot.concept_pipeline.baseline_verbalization import (
    analyze_verbalization_on_baseline_for_stage,
)
from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.data_consistency import (
    validate_concepts_at_stage_start,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.stage import (
    compute_concepts_in_last_stage,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.variation_verbalization import (
    analyze_verbalization_on_variations_for_stage,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)


class _FailIfCalledVerbalizationDetector:
    async def is_verbalized_baseline_batch(self, *_args, **_kwargs):
        raise AssertionError("LLM-based baseline verbalization should not be called")

    async def is_verbalized_variations_batch(self, *_args, **_kwargs):
        raise AssertionError("LLM-based variation verbalization should not be called")


@pytest.mark.anyio
async def test_baseline_verbalization_carryover_synthetic(tmp_path: Path) -> None:
    # Deterministic IDs
    concept_id = ConceptId("00000000-0000-0000-0000-000000000101")
    input_a = InputId("00000000-0000-0000-0000-00000000010a")
    input_b = InputId("00000000-0000-0000-0000-00000000010b")
    rid_a = ResponseId("00000000-0000-0000-0000-0000000001a1")
    rid_b = ResponseId("00000000-0000-0000-0000-0000000001b1")

    concept = Concept(
        title="Synthetic — Baseline",
        verbalization_check_guide="Irrelevant (no LLM calls in this test).",
        removal_action="remove",
        addition_action="add",
        id=concept_id,
    )

    dataset = ConceptPipelineDataset(
        dataset_name="synthetic",
        input_template="Task: {x}",
        input_parameters={},
        varying_input_param_name="x",
        varying_inputs={input_a: "A", input_b: "B"},
        sanitized_varying_inputs={input_a: "A", input_b: "B"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations=None,
    )

    stage0 = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a, input_b]},
        concepts_at_stage_start=[concept_id],
        concept_verbalization_on_baseline_responses={
            concept_id: {
                input_a: {
                    rid_a: VerbalizationCheckResult(verbalized=False, witness="")
                },
                input_b: {
                    rid_b: VerbalizationCheckResult(verbalized=False, witness="")
                },
            }
        },
        concept_ids_unverbalized_on_baseline=[concept_id],
    )

    stage1 = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a, input_b]},
        concepts_at_stage_start=[concept_id],
        concept_verbalization_on_baseline_responses=None,
        concept_ids_unverbalized_on_baseline=None,
    )

    result = ConceptPipelineResult(
        experiment_key="synthetic",
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
        significance_test="fisher",
        baseline_responses_by_input={
            input_a: {rid_a: "resp A"},
            input_b: {rid_b: "resp B"},
        },
        stages=[stage0, stage1],
    )

    detector = _FailIfCalledVerbalizationDetector()
    await analyze_verbalization_on_baseline_for_stage(
        result,
        stage1,
        dataset=dataset,
        output_dir=tmp_path,
        baseline_verbalization_threshold=0.25,
        verbalization_detector=detector,  # type: ignore[arg-type]
    )

    # Stage>0 baseline verbalization is skipped; we only carry over the filtering decision.
    assert stage1.concept_verbalization_on_baseline_responses is None
    assert stage1.concept_ids_unverbalized_on_baseline == [concept_id]
    assert stage1.concept_ids_unverbalized_on_baseline == [concept_id]


@pytest.mark.anyio
async def test_variation_verbalization_carryover_synthetic(tmp_path: Path) -> None:
    concept_id = ConceptId("00000000-0000-0000-0000-000000000201")
    input_a = InputId("00000000-0000-0000-0000-00000000020a")
    input_b = InputId("00000000-0000-0000-0000-00000000020b")
    pair_id = VariationPairId("00000000-0000-0000-0000-0000000002f1")

    rid_a_pos = ResponseId("00000000-0000-0000-0000-0000000002a1")
    rid_a_neg = ResponseId("00000000-0000-0000-0000-0000000002a2")
    rid_b_pos = ResponseId("00000000-0000-0000-0000-0000000002b1")
    rid_b_neg = ResponseId("00000000-0000-0000-0000-0000000002b2")

    concept = Concept(
        title="Synthetic — Variation",
        verbalization_check_guide="Irrelevant (no LLM calls in this test).",
        removal_action="remove",
        addition_action="add",
        id=concept_id,
    )

    dataset = ConceptPipelineDataset(
        dataset_name="synthetic",
        input_template="Task: {x}",
        input_parameters={},
        varying_input_param_name="x",
        varying_inputs={input_a: "A", input_b: "B"},
        sanitized_varying_inputs={input_a: "A", input_b: "B"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations={
            concept_id: {
                input_a: {pair_id: VariationPair(positive="pos A", negative="neg A")},
                input_b: {pair_id: VariationPair(positive="pos B", negative="neg B")},
            }
        },
    )

    stage0_bias = ConceptBiasTestResult(
        responses_by_input={
            input_a: {
                pair_id: VariationPairResponses(
                    positive_responses={rid_a_pos: "pos A"},
                    negative_responses={rid_a_neg: "neg A"},
                    positive_acceptances={rid_a_pos: 1},
                    negative_acceptances={rid_a_neg: 0},
                )
            },
            input_b: {
                pair_id: VariationPairResponses(
                    positive_responses={rid_b_pos: "pos B"},
                    negative_responses={rid_b_neg: "neg B"},
                    positive_acceptances={rid_b_pos: 1},
                    negative_acceptances={rid_b_neg: 0},
                )
            },
        },
        flipped_variation_pairs={
            input_a: {pair_id: True},
            input_b: {pair_id: True},
        },
    )

    stage0 = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a, input_b]},
        concepts_at_stage_start=[concept_id],
        variation_bias_results={concept_id: stage0_bias},
        significant_concepts=[concept_id],
        concept_verbalization_on_variation_responses={
            concept_id: {
                input_a: {
                    pair_id: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_a_pos: VerbalizationCheckResult(False, "")
                        },
                        negative_variation_responses_verbalizations={
                            rid_a_neg: VerbalizationCheckResult(False, "")
                        },
                    )
                },
                input_b: {
                    pair_id: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_b_pos: VerbalizationCheckResult(False, "")
                        },
                        negative_variation_responses_verbalizations={
                            rid_b_neg: VerbalizationCheckResult(False, "")
                        },
                    )
                },
            }
        },
    )

    stage1 = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a, input_b]},
        concepts_at_stage_start=[concept_id],
        variation_bias_results=None,
        significant_concepts=[concept_id],
        concept_verbalization_on_variation_responses=None,
    )

    # Construct a real ConceptBiasTestResult for stage 1 by reusing the stage-0 one.
    # This mirrors the intended carry-over behavior: stable ResponseIds for shared inputs.
    stage1.variation_bias_results = stage0.variation_bias_results

    result = ConceptPipelineResult(
        experiment_key="synthetic",
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
        significance_test="fisher",
        stages=[stage0, stage1],
    )

    detector = _FailIfCalledVerbalizationDetector()
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage1,
        dataset=dataset,
        output_dir=tmp_path,
        verbalization_detector=detector,  # type: ignore[arg-type]
    )

    assert stage1.concept_verbalization_on_variation_responses is not None
    assert stage1.concept_verbalization_on_variation_responses[concept_id][input_a][
        pair_id
    ].positive_variation_responses_verbalizations.keys() == {rid_a_pos}
    assert stage1.concept_verbalization_on_variation_responses[concept_id][input_b][
        pair_id
    ].negative_variation_responses_verbalizations.keys() == {rid_b_neg}


def test_stopped_concepts_not_carried_to_next_stage() -> None:
    """Futility-stopped and early-stopped concepts must not appear in later stages."""

    concept_futility = ConceptId("00000000-0000-0000-0000-00000000f001")
    concept_early = ConceptId("00000000-0000-0000-0000-00000000e001")
    input_a = InputId("00000000-0000-0000-0000-0000000000aa")

    concept_a = Concept(
        title="Synthetic — Futility",
        verbalization_check_guide="",
        removal_action="remove",
        addition_action="add",
        id=concept_futility,
    )
    concept_b = Concept(
        title="Synthetic — Early",
        verbalization_check_guide="",
        removal_action="remove",
        addition_action="add",
        id=concept_early,
    )

    dataset = ConceptPipelineDataset(
        dataset_name="synthetic",
        input_template="Task: {x}",
        input_parameters={},
        varying_input_param_name="x",
        varying_inputs={input_a: "A"},
        sanitized_varying_inputs={input_a: "A"},
        concepts=[concept_a, concept_b],
        deduplicated_concepts=[concept_a, concept_b],
        variations=None,
    )

    stage0 = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a]},
        concepts_at_stage_start=[concept_futility, concept_early],
        concepts_at_stage_end=[concept_futility, concept_early],
        futility_stopped_concepts=[concept_futility],
        early_stopped_concepts=[concept_early],
    )

    result = ConceptPipelineResult(
        experiment_key="synthetic",
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
        significance_test="fisher",
        stages=[stage0],
    )

    carried = compute_concepts_in_last_stage(result, 1, dataset=dataset)
    assert carried == []

    stage1 = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a]},
        concepts_at_stage_start=[],
    )
    result.stages.append(stage1)  # type: ignore
    validate_concepts_at_stage_start(dataset, result, stage1)
