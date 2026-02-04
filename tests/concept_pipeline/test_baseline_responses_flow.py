from pathlib import Path

import pytest

from biases_in_the_blind_spot.concept_pipeline.baseline_responses import (
    prepare_pre_filtered_baseline_and_filter_inputs_if_needed,
)
from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)


class _StubResponsesGenerator(ResponsesGenerator):
    def __init__(self, outputs: list[str]):
        super().__init__()
        self._outputs = outputs
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return list(self._outputs)


class _StubBiasTester(BiasTester):
    def __init__(self, accepts: list[int | None]):
        super().__init__(responses_generator=None, parse_response_fn=lambda *_: None)  # type: ignore[arg-type]
        self._accepts = accepts
        self.calls = 0

    def calculate_acceptances_for_responses(
        self, input_id: InputId, responses: list[str]
    ):
        self.calls += 1
        assert len(responses) == len(self._accepts)
        return list(self._accepts)


def _dataset(input_id: InputId) -> ConceptPipelineDataset:
    return ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{vary}",
        input_parameters={"vary": "x"},
        varying_input_param_name="vary",
        varying_inputs={input_id: "text"},
        sanitized_varying_inputs={input_id: "text"},
    )


def test_prepare_prefilter_populates_when_missing(tmp_path: Path):
    iid = InputId()
    dataset = _dataset(iid)
    result = ConceptPipelineResult(
        filtered_varying_inputs=[iid],
        baseline_responses_by_input={},
        baseline_acceptances_by_input={},
        stages=[],
    )
    resp_gen = _StubResponsesGenerator(["r1", "r2"])
    bias_tester = _StubBiasTester([1, 0])

    prepare_pre_filtered_baseline_and_filter_inputs_if_needed(
        result,
        dataset=dataset,
        responses_generator=resp_gen,
        bias_tester=bias_tester,
        input_prefilter=None,
        output_dir=tmp_path,
        n_baseline_responses_pre_filter_per_input=2,
        n_baseline_responses_per_input=2,
    )

    assert resp_gen.calls == 1
    assert bias_tester.calls == 1
    assert result.pre_filtered_baseline_responses_by_input is not None
    assert result.pre_filtered_baseline_acceptances_by_input is not None
    assert len(result.pre_filtered_baseline_responses_by_input[iid]) == 2


def test_prepare_prefilter_existing_mismatch_raises(tmp_path: Path):
    iid = InputId()
    dataset = _dataset(iid)
    result = ConceptPipelineResult(
        pre_filtered_baseline_responses_by_input={iid: {ResponseId(): "r"}},
        pre_filtered_baseline_acceptances_by_input=None,
        stages=[],
    )
    resp_gen = _StubResponsesGenerator([])
    bias_tester = _StubBiasTester([])

    with pytest.raises(ValueError):
        prepare_pre_filtered_baseline_and_filter_inputs_if_needed(
            result,
            dataset=dataset,
            responses_generator=resp_gen,
            bias_tester=bias_tester,
            input_prefilter=None,
            output_dir=tmp_path,
            n_baseline_responses_pre_filter_per_input=1,
            n_baseline_responses_per_input=1,
        )
