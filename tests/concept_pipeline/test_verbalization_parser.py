from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    default_verbalization_parser,
)


def test_parser_yes_with_witness():
    out = default_verbalization_parser(
        "<verbalized>YES</verbalized><witness>snippet</witness>"
    )
    assert isinstance(out, VerbalizationCheckResult)
    assert out.verbalized is True
    assert out.witness == "snippet"


def test_parser_no_without_witness():
    out = default_verbalization_parser("<verbalized>NO</verbalized>")
    assert isinstance(out, VerbalizationCheckResult)
    assert out.verbalized is False
    assert out.witness == ""


def test_parser_missing_tags_returns_none():
    assert default_verbalization_parser("junk") is None


def test_parser_wrong_value_returns_none():
    assert default_verbalization_parser("<verbalized>MAYBE</verbalized>") is None
