from pathlib import Path

import pytest

from biases_in_the_blind_spot.concept_pipeline import variation_bias
from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
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
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)
from biases_in_the_blind_spot.concept_pipeline.statistics import check_futility_stopping
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
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)


@pytest.mark.anyio
async def test_variation_bias_carryover_prevents_response_id_mismatch(
    tmp_path: Path,
) -> None:
    """Synthetic repro for the stage-0 -> stage-1 mismatch path.

    This triggers the same check that previously raised:
      ValueError: Positive variation responses mismatch for concept ..., input ..., pair ...

    but in a fully synthetic setup (no disk fixtures, no LLM calls). The test passes
    iff stage-1 variation-bias reuses stage-0 response ids for overlapping inputs.
    """

    # --- Synthetic IDs (fixed UUIDs to make the test deterministic) ---
    concept_id = ConceptId("00000000-0000-0000-0000-000000000001")
    input_a = InputId("00000000-0000-0000-0000-00000000000a")
    input_b = InputId("00000000-0000-0000-0000-00000000000b")
    pair_id = VariationPairId("00000000-0000-0000-0000-0000000000f1")

    rid_a_pos = ResponseId("00000000-0000-0000-0000-00000000a0f1")
    rid_a_neg = ResponseId("00000000-0000-0000-0000-00000000a0e1")
    rid_b_pos = ResponseId("00000000-0000-0000-0000-00000000b0f1")
    rid_b_neg = ResponseId("00000000-0000-0000-0000-00000000b0e1")

    # --- Minimal dataset with variations present for the stage inputs ---
    concept = Concept(
        title="Synthetic — Concept",
        verbalization_check_guide="Look for any mention of the concept.",
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

    # --- Stage 0: has variation-bias responses and variation-verbalization keyed by those response ids ---
    stage0_bias = ConceptBiasTestResult(
        responses_by_input={
            input_a: {
                pair_id: VariationPairResponses(
                    positive_responses={rid_a_pos: "pos resp A"},
                    negative_responses={rid_a_neg: "neg resp A"},
                    positive_acceptances={rid_a_pos: 1},
                    negative_acceptances={rid_a_neg: 0},
                )
            },
            input_b: {
                pair_id: VariationPairResponses(
                    positive_responses={rid_b_pos: "pos resp B"},
                    negative_responses={rid_b_neg: "neg resp B"},
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
        concept_ids_unverbalized_on_baseline=[concept_id],
        variation_bias_results={concept_id: stage0_bias},
        significant_concepts=[concept_id],
        concept_verbalization_on_variation_responses={
            concept_id: {
                input_a: {
                    pair_id: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_a_pos: VerbalizationCheckResult(
                                verbalized=False, witness=""
                            )
                        },
                        negative_variation_responses_verbalizations={
                            rid_a_neg: VerbalizationCheckResult(
                                verbalized=False, witness=""
                            )
                        },
                    )
                },
                input_b: {
                    pair_id: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_b_pos: VerbalizationCheckResult(
                                verbalized=False, witness=""
                            )
                        },
                        negative_variation_responses_verbalizations={
                            rid_b_neg: VerbalizationCheckResult(
                                verbalized=False, witness=""
                            )
                        },
                    )
                },
            }
        },
    )

    # --- Stage 1: starts "fresh" (no bias results, no variation-verbalization cache) ---
    stage1 = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_a, input_b]},
        concepts_at_stage_start=[concept_id],
        concept_ids_unverbalized_on_baseline=[concept_id],
        variation_bias_results=None,
        significant_concepts=None,
        concept_verbalization_on_variation_responses=None,
    )

    result = ConceptPipelineResult(
        experiment_key="synthetic",
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
        significance_test="fisher",
        filtered_varying_inputs=[input_a, input_b],
        futility_stop_power_threshold=0.10,
        stages=[stage0, stage1],
    )

    # --- Exercise the carry-over fix: stage 1 bias results should be reused for overlapping inputs ---
    dummy_generator = ResponsesGenerator(model_name="dummy")
    dummy_bias_tester = BiasTester(
        responses_generator=dummy_generator,
        parse_response_fn=lambda _input_id, _resp: None,
    )
    variation_bias.test_variations_bias_for_stage(
        result,
        stage1,
        dataset=dataset,
        bias_tester=dummy_bias_tester,
        responses_generator=dummy_generator,
        output_dir=tmp_path,
    )

    # Pipeline contract: futility stopping is the authority for significant_concepts.
    stage1.stage_significant_concepts_p_value_alpha = 0.05
    check_futility_stopping(dataset, result, stage1)

    # --- Exercise the exact mismatch-check path in variation verbalization (should not raise) ---
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage1,
        dataset=dataset,
        output_dir=tmp_path,
        verbalization_detector=VerbalizationDetector(),
    )
