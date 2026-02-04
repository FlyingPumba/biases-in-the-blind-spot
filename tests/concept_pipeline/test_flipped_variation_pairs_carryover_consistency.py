# isort: skip_file

from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId

import biases_in_the_blind_spot.concept_pipeline.variation_bias as variation_bias
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from tests.concept_pipeline.helpers import make_concept


class _EchoResponsesGenerator:
    model_name = "stub"

    def generate(
        self, _tmpl: str, input_parameters_list: list[dict[str, str]], **_: object
    ):
        # BiasTester sets variating_input_parameter to the variation string; echo it back.
        # This lets parse_response_fn decide acceptance deterministically.
        return [params["vary"] for params in input_parameters_list]


def _make_dataset_one_concept_three_inputs(
    *,
    concept_id: ConceptId,
    inputs: list[InputId],
    pair_id: VariationPairId,
) -> ConceptPipelineDataset:
    assert len(inputs) == 3
    concept = make_concept(concept_id)
    variations = {
        concept_id: {
            inputs[0]: {pair_id: VariationPair("pos_A", "neg_A")},
            inputs[1]: {pair_id: VariationPair("pos_B", "neg_B")},
            inputs[2]: {pair_id: VariationPair("pos_C", "neg_C")},
        }
    }
    return ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{vary}",
        input_parameters={},
        varying_input_param_name="vary",
        varying_inputs=dict.fromkeys(inputs, "x"),
        sanitized_varying_inputs=dict.fromkeys(inputs, "x"),
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations=variations,
    )


def _make_stage(
    stage_idx: int, input_ids: list[InputId], *, concept_id: ConceptId
) -> StageResults:
    return StageResults(
        stage_idx=stage_idx,
        k_inputs_per_representative_cluster=len(input_ids),
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): list(input_ids)},
        concepts_at_stage_start=[concept_id],
        concept_ids_unverbalized_on_baseline=[concept_id],
        variation_bias_results=None,
        significant_concepts=None,
    )


def _compute_flips_for_result(
    res,
) -> dict[InputId, dict[VariationPairId, bool]]:
    """Compute flipped map from responses_by_input (test helper).

    Assumption: exactly one positive and one negative response per pair.
    """
    responses_by_input = res.responses_by_input
    out: dict[InputId, dict[VariationPairId, bool]] = {}
    for input_id, by_pair in responses_by_input.items():
        per_input: dict[VariationPairId, bool] = {}
        for pair_id, pair_responses in by_pair.items():
            assert isinstance(pair_responses, VariationPairResponses)
            assert len(pair_responses.positive_acceptances) == 1
            assert len(pair_responses.negative_acceptances) == 1
            per_input[pair_id] = pair_responses.has_flipped_acceptance()
        out[input_id] = per_input
    return out


def test_flipped_variation_pairs_stage1_superset_inputs_is_consistent(tmp_path):
    """Regression: Stage 1 carry-over + new inputs must keep flipped map consistent.

    Stage 0 has inputs A,B. Stage 1 has A,B,C (adds new input C).
    This must not crash and must produce a flipped_variation_pairs map matching responses_by_input.
    """
    concept_id = ConceptId()
    inputs = [InputId(), InputId(), InputId()]
    input_a, input_b, input_c = inputs
    pair_id = VariationPairId()
    dataset = _make_dataset_one_concept_three_inputs(
        concept_id=concept_id, inputs=inputs, pair_id=pair_id
    )

    bias_tester = BiasTester(
        responses_generator=_EchoResponsesGenerator(),  # type: ignore[arg-type]
        parse_response_fn=lambda _iid, resp: 1 if "pos" in resp else 0,
        variating_input_parameter="vary",
    )

    stage0 = _make_stage(0, [input_a, input_b], concept_id=concept_id)
    stage1 = _make_stage(1, [input_a, input_b, input_c], concept_id=concept_id)
    result = ConceptPipelineResult(
        stages=[stage0, stage1], filtered_varying_inputs=inputs
    )

    # First run stage 0 to materialize bias results + flipped_variation_pairs.
    variation_bias.test_variations_bias_for_stage(
        result,
        stage0,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,
        output_dir=tmp_path,
    )
    assert stage0.variation_bias_results is not None
    stage0_res = stage0.variation_bias_results[concept_id]
    assert stage0_res.flipped_variation_pairs is not None

    variation_bias.test_variations_bias_for_stage(
        result,
        stage1,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,
        output_dir=tmp_path,
    )
    assert stage1.variation_bias_results is not None
    stage1_res = stage1.variation_bias_results[concept_id]
    assert stage1_res.flipped_variation_pairs is not None

    computed = _compute_flips_for_result(stage1_res)
    assert stage1_res.flipped_variation_pairs == computed


def test_flipped_variation_pairs_stage1_subset_inputs_is_consistent(tmp_path):
    """Regression: Stage 1 subset inputs must not keep stale flipped-map keys.

    Stage 0 has inputs A,B. Stage 1 has B,C (drops A, adds C).
    This must not crash and must produce a flipped_variation_pairs map matching responses_by_input.
    """
    concept_id = ConceptId()
    inputs = [InputId(), InputId(), InputId()]
    input_a, input_b, input_c = inputs
    pair_id = VariationPairId()
    dataset = _make_dataset_one_concept_three_inputs(
        concept_id=concept_id, inputs=inputs, pair_id=pair_id
    )

    bias_tester = BiasTester(
        responses_generator=_EchoResponsesGenerator(),  # type: ignore[arg-type]
        parse_response_fn=lambda _iid, resp: 1 if "pos" in resp else 0,
        variating_input_parameter="vary",
    )

    stage0 = _make_stage(0, [input_a, input_b], concept_id=concept_id)
    # Stage 1 uses only B,C (drops A) and adds C.
    stage1 = _make_stage(1, [input_b, input_c], concept_id=concept_id)
    result = ConceptPipelineResult(
        stages=[stage0, stage1], filtered_varying_inputs=inputs
    )

    variation_bias.test_variations_bias_for_stage(
        result,
        stage0,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,
        output_dir=tmp_path,
    )
    assert stage0.variation_bias_results is not None
    assert stage0.variation_bias_results[concept_id].flipped_variation_pairs is not None

    variation_bias.test_variations_bias_for_stage(
        result,
        stage1,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,
        output_dir=tmp_path,
    )
    assert stage1.variation_bias_results is not None
    stage1_res = stage1.variation_bias_results[concept_id]
    assert stage1_res.flipped_variation_pairs is not None
    computed = _compute_flips_for_result(stage1_res)
    assert stage1_res.flipped_variation_pairs == computed
