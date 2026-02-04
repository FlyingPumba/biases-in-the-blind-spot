from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)


def test_variation_pair_verbalization_stores_maps():
    rid_pos = ResponseId()
    rid_neg = ResponseId()
    vpv = VariationPairVerbalization(
        positive_variation_responses_verbalizations={
            rid_pos: VerbalizationCheckResult(True, "w")
        },
        negative_variation_responses_verbalizations={
            rid_neg: VerbalizationCheckResult(False, "")
        },
    )
    assert vpv.positive_variation_responses_verbalizations[rid_pos].verbalized
    assert vpv.negative_variation_responses_verbalizations[rid_neg].verbalized is False
