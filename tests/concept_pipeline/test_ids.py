import pytest

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId


def test_uuid_ids_generate_and_validate():
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    vid = VariationPairId()
    # Round-trip string is preserved
    assert ConceptId(cid) == cid
    assert InputId(iid) == iid
    assert ResponseId(rid) == rid
    assert VariationPairId(vid) == vid


@pytest.mark.parametrize("cls", [ConceptId, InputId, ResponseId, VariationPairId])
def test_uuid_ids_reject_invalid(cls):
    with pytest.raises(ValueError):
        cls("not-a-uuid")


def test_cluster_id_parses_int_and_string():
    assert ClusterId(3) == 3
    assert ClusterId("4") == 4
    with pytest.raises(ValueError):
        ClusterId("not-int")
