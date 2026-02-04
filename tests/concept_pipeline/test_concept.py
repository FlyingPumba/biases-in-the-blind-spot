from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId


def test_concept_defaults_and_id():
    cid = ConceptId()
    c = Concept(
        title="Title—Cat",
        verbalization_check_guide="guide",
        removal_action="rem",
        addition_action="add",
        id=cid,
    )
    assert c.id == cid
    assert "guide" in c.verbalization_check_guide
