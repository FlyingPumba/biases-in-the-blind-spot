from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)


def test_concept_bias_test_result_counts_and_flatten():
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()
    vpr = VariationPairResponses(
        positive_responses={rid_pos: "p"},
        negative_responses={rid_neg: "n"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    res = ConceptBiasTestResult(responses_by_input={iid: {pid: vpr}})
    assert res.num_variations == 1
    flat = res.flatten_acceptances()
    assert flat == [1, 0]
