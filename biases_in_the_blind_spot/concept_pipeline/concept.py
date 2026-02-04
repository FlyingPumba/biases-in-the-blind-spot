from dataclasses import dataclass

from dataclass_wizard import JSONWizard

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId


@dataclass
class Concept(JSONWizard):
    """A concept hypothesis with title and a guide to check verbalization."""

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    title: str
    verbalization_check_guide: str
    removal_action: str
    addition_action: str
    id: ConceptId = ConceptId()
