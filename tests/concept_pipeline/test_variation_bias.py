import pytest

import biases_in_the_blind_spot.concept_pipeline.variation_bias as vb
from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from tests.concept_pipeline.helpers import make_concept

# Prevent pytest from collecting the helper in variation_bias module
vb.test_variations_bias_for_stage.__test__ = False


def _dataset_with_variations(cid: ConceptId, iid: InputId, pid: VariationPairId):
    concept = make_concept(cid)
    ds = ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{vary}",
        input_parameters={iid: {"vary": "text"}},  # type: ignore[arg-type]
        varying_input_param_name="vary",
        varying_inputs={iid: "text"},
        sanitized_varying_inputs={iid: "text"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        variations={cid: {iid: {pid: VariationPair("p", "n")}}},
    )
    # Attach expected attrs dynamically to satisfy variation_bias
    ds.concepts_by_id = {cid: concept}  # type: ignore[attr-defined]
    ds.variation_template = "{vary}"  # type: ignore[attr-defined]
    return ds


def _concept_bias_result(iid: InputId, pid: VariationPairId):
    rid_pos = ResponseId()
    rid_neg = ResponseId()
    vpr = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    return ConceptBiasTestResult(responses_by_input={iid: {pid: vpr}})


def test_variation_bias_generation_populates_and_stats(tmp_path):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = _dataset_with_variations(cid, iid, pid)
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        variation_bias_results=None,
        significant_concepts=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
        significance_test="fisher",
    )

    class _RespGen:
        model_name = "stub"

        def generate(
            self,
            input_template: str,
            input_parameters_list: list[dict[str, str]],
            **_: object,
        ):
            return ["resp_pos"] * len(input_parameters_list)

    def _parse(_iid: InputId, resp: str) -> int | None:
        return 1 if "pos" in resp else 0

    bias_tester = BiasTester(
        responses_generator=_RespGen(),  # type: ignore[arg-type]
        parse_response_fn=_parse,  # type: ignore[arg-type]
    )
    vb.test_variations_bias_for_stage(
        result,
        stage,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,  # type: ignore[arg-type]
        output_dir=tmp_path,
    )
    assert stage.variation_bias_results is not None
    res = stage.variation_bias_results[cid]
    stats = res.statistics_positive_vs_negative
    assert stats is not None
    assert stats.get("fisher_p_value") is not None
    assert res.flipped_variation_pairs is not None
    assert res.flipped_variation_pairs[iid][pid] is False


def test_variation_bias_uses_variation_values(tmp_path):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = _dataset_with_variations(cid, iid, pid)
    # Add a second input to ensure we merge per-input results for the same concept
    iid2 = InputId()
    dataset.input_parameters[iid2] = {"vary": "text2"}  # type: ignore[index]
    dataset.varying_inputs[iid2] = "text2"  # type: ignore[index]
    dataset.sanitized_varying_inputs[iid2] = "text2"  # type: ignore[index]
    dataset.variations[cid][iid2] = {pid: VariationPair("p2", "n2")}  # type: ignore[index]
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=2,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid, iid2]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        variation_bias_results=None,
        significant_concepts=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid, iid2],
        significance_test="fisher",
    )

    class _RespGen:
        model_name = "stub"

        def __init__(self):
            self.calls: list[tuple[str, list[dict[str, str]]]] = []

        def generate(
            self,
            input_template: str,
            input_parameters_list: list[dict[str, str]],
            **_: object,
        ):
            self.calls.append((input_template, input_parameters_list))
            return [
                params[dataset.varying_input_param_name]
                for params in input_parameters_list
            ]

    def _parse(_iid: InputId, resp: str) -> int | None:
        return 1 if "pos" in resp else 0

    responses_generator = _RespGen()
    bias_tester = BiasTester(
        responses_generator=responses_generator,  # type: ignore[arg-type]
        parse_response_fn=_parse,  # type: ignore[arg-type]
    )
    bias_tester.variating_input_parameter = dataset.varying_input_param_name

    vb.test_variations_bias_for_stage(
        result,
        stage,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=responses_generator,  # type: ignore[arg-type]
        output_dir=tmp_path,
    )

    # Single batched call with all prompts (pos+neg for each input)
    assert len(responses_generator.calls) == 1
    template, params_batch = responses_generator.calls[0]
    assert template == dataset.input_template
    assert len(params_batch) == 4  # 2 inputs × (pos+neg)
    vary_values = {params[dataset.varying_input_param_name] for params in params_batch}
    expected_vary_values = {
        dataset.variations[cid][iid][pid].positive,  # type: ignore[index]
        dataset.variations[cid][iid][pid].negative,  # type: ignore[index]
        dataset.variations[cid][iid2][pid].positive,  # type: ignore[index]
        dataset.variations[cid][iid2][pid].negative,  # type: ignore[index]
    }
    assert vary_values == expected_vary_values
    # Ensure both inputs are stored in the result
    assert stage.variation_bias_results is not None
    res = stage.variation_bias_results[cid]
    assert set(res.responses_by_input.keys()) == {iid, iid2}
    assert res.flipped_variation_pairs is not None
    assert res.flipped_variation_pairs[iid][pid] is False
    assert res.flipped_variation_pairs[iid2][pid] is False


def test_variation_bias_existing_results_noop(tmp_path):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = _dataset_with_variations(cid, iid, pid)
    bias_result = _concept_bias_result(iid, pid)
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        variation_bias_results={cid: bias_result},
        significant_concepts=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
    )

    # save_result / plot are already stubbed in conftest
    class _RG:
        model_name = "stub"

        def generate(self, *_: object, **__: object):
            return []

    bias_tester = BiasTester(
        responses_generator=_RG(),  # type: ignore[arg-type]
        parse_response_fn=lambda _i, r: 1,
    )
    vb.test_variations_bias_for_stage(
        result,
        stage,
        dataset=dataset,
        bias_tester=bias_tester,
        responses_generator=bias_tester.responses_generator,  # type: ignore[arg-type]
        output_dir=tmp_path,
    )
    assert stage.variation_bias_results == {cid: bias_result}
    assert bias_result.flipped_variation_pairs is not None
    assert bias_result.flipped_variation_pairs[iid][pid] is True


def test_variation_bias_raises_on_missing_pairs(tmp_path):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    pid_extra = VariationPairId()
    dataset = _dataset_with_variations(cid, iid, pid)
    dataset.variations[cid][iid][pid_extra] = VariationPair("p2", "n2")  # type: ignore[index]
    bias_result = _concept_bias_result(iid, pid)
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        variation_bias_results={cid: bias_result},
        significant_concepts=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
    )

    class _RG:
        model_name = "stub"

        def generate(self, *_: object, **__: object):
            return []

    bias_tester = BiasTester(
        responses_generator=_RG(),  # type: ignore[arg-type]
        parse_response_fn=lambda _i, r: 1,
    )

    with pytest.raises(ValueError):
        vb.test_variations_bias_for_stage(
            result,
            stage,
            dataset=dataset,
            bias_tester=bias_tester,
            responses_generator=bias_tester.responses_generator,  # type: ignore[arg-type]
            output_dir=tmp_path,
        )


def test_variation_bias_raises_on_empty_response(tmp_path):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = _dataset_with_variations(cid, iid, pid)
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        variation_bias_results=None,
        significant_concepts=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[iid],
        significance_test="fisher",
    )

    class _RespGen:
        model_name = "stub"

        def generate(
            self,
            input_template: str,
            input_parameters_list: list[dict[str, str]],
            **_: object,
        ):
            return [""] * len(input_parameters_list)

    def _parse(_iid: InputId, resp: str) -> int | None:
        return 1 if resp else None

    bias_tester = BiasTester(
        responses_generator=_RespGen(),  # type: ignore[arg-type]
        parse_response_fn=_parse,  # type: ignore[arg-type]
    )

    with pytest.raises(BiasTester.EmptyResponseError):
        vb.test_variations_bias_for_stage(
            result,
            stage,
            dataset=dataset,
            bias_tester=bias_tester,
            responses_generator=bias_tester.responses_generator,  # type: ignore[arg-type]
            output_dir=tmp_path,
        )
