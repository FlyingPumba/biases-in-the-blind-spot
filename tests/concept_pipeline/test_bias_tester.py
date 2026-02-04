from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId


def _parse_response_fn(_iid: InputId, resp: str) -> int | None:
    return 1 if "YES" in resp else 0


def test_bias_tester_config_and_empty_indices():
    tester = BiasTester(responses_generator=None, parse_response_fn=_parse_response_fn)  # type: ignore[arg-type]
    cfg = tester.config
    assert "variations_bias_temperature" in cfg
    assert tester._empty_response_indices(["", "a"]) == [0]


def test_calculate_acceptances_for_responses():
    tester = BiasTester(responses_generator=None, parse_response_fn=_parse_response_fn)  # type: ignore[arg-type]
    iid = InputId()
    accepts = tester.calculate_acceptances_for_responses(iid, ["YES", "NO", "other"])
    assert accepts == [1, 0, 0]
