import pytest

from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair


def test_get_variation_by_side():
    vp = VariationPair(positive="yes", negative="no")
    assert vp.get_variation_by_side("positive") == "yes"
    assert vp.get_variation_by_side("negative") == "no"
    with pytest.raises(ValueError):
        vp.get_variation_by_side("other")
