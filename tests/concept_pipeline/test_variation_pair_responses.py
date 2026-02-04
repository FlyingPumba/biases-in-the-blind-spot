from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)


def test_has_flipped_acceptance_true_when_diff():
    rid = ResponseId()
    vpr = VariationPairResponses(
        positive_responses={rid: "p"},
        negative_responses={rid: "n"},
        positive_acceptances={rid: 1},
        negative_acceptances={rid: 0},
    )
    assert vpr.has_flipped_acceptance() is True


def test_has_flipped_acceptance_false_when_same():
    rid = ResponseId()
    vpr = VariationPairResponses(
        positive_responses={rid: "p"},
        negative_responses={rid: "n"},
        positive_acceptances={rid: 1},
        negative_acceptances={rid: 1},
    )
    assert vpr.has_flipped_acceptance() is False


def test_has_flipped_acceptance_ignores_none():
    rid = ResponseId()
    vpr = VariationPairResponses(
        positive_responses={rid: "p"},
        negative_responses={rid: "n"},
        positive_acceptances={rid: None},
        negative_acceptances={rid: None},
    )
    assert vpr.has_flipped_acceptance() is False
