from dataclasses import dataclass

from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)


@dataclass
class VariationPairVerbalization:
    positive_variation_responses_verbalizations: dict[
        ResponseId, VerbalizationCheckResult
    ]
    negative_variation_responses_verbalizations: dict[
        ResponseId, VerbalizationCheckResult
    ]
