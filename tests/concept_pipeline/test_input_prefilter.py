from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.input_prefilter import InputPrefilter
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId


def test_input_prefilter_defaults_and_config():
    pf = InputPrefilter(max_inputs=2)
    cfg = pf.config
    assert cfg["max_inputs"] == 2


def test_input_prefilter_filter_inputs():
    pf = InputPrefilter(max_inputs=2)
    a, b, c = InputId(), InputId(), InputId()
    rid1, rid2 = ResponseId(), ResponseId()
    keep = pf.filter_inputs(
        [a, b, c],
        {
            a: {rid1: 1, rid2: 1},
            b: {rid1: 0, rid2: 1},
            c: {rid1: None, rid2: None},
        },
    )
    assert keep[:2] == [b, a]  # b has variance, a lower, c zero variance
