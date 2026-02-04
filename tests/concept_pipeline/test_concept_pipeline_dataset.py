import pytest

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from tests.concept_pipeline.helpers import make_concept, make_dataset


def test_get_concept_title():
    cid = ConceptId()
    iid = InputId()
    ds = make_dataset(cid, iid)
    assert ds.get_concept_title(cid) == make_concept(cid).title
    with pytest.raises(KeyError):
        ds.get_concept_title(ConceptId())


def test_get_input_params():
    cid = ConceptId()
    iid = InputId()
    ds = make_dataset(cid, iid)
    params = ds.get_input_params(iid)
    assert params["vary"] == "text"
    assert ds.varying_input_param_name in params
    with pytest.raises(AssertionError):
        ds.get_input_params(InputId())
