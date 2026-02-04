from dataclasses import dataclass

from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId


@dataclass
class VariationPairResponses:
    positive_responses: dict[ResponseId, str]
    negative_responses: dict[ResponseId, str]
    positive_acceptances: dict[ResponseId, int | None]
    negative_acceptances: dict[ResponseId, int | None]

    def has_flipped_acceptance(self) -> bool:
        pos_vals = list(self.positive_acceptances.values())
        neg_vals = list(self.negative_acceptances.values())
        assert len(pos_vals) == len(neg_vals)
        assert len(pos_vals) <= 1, (
            "Only one variation pair is supported for now. Otherwise we need to implement something more sophisticated to understand the flipping of acceptance."
        )
        for pos, neg in zip(pos_vals, neg_vals, strict=True):
            if pos is None or neg is None:
                continue
            if pos != neg:
                return True
        return False
