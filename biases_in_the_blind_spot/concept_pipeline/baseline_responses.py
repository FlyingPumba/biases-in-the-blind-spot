from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.input_prefilter import InputPrefilter
from biases_in_the_blind_spot.concept_pipeline.pipeline_persistence import (
    get_result_path,
    save_result,
)
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)


def prepare_pre_filtered_baseline_and_filter_inputs_if_needed(
    result: ConceptPipelineResult,
    *,
    dataset: ConceptPipelineDataset,
    responses_generator: ResponsesGenerator,
    bias_tester: BiasTester,
    input_prefilter: InputPrefilter | None,
    output_dir: Path,
    n_baseline_responses_pre_filter_per_input: int,
    n_baseline_responses_per_input: int,
) -> None:
    """Ensure pre-filter baseline data exists, then filter inputs once."""
    assert dataset.sanitized_varying_inputs is not None
    effective_n_per = (
        n_baseline_responses_pre_filter_per_input
        if input_prefilter is not None
        else n_baseline_responses_per_input
    )
    assert isinstance(effective_n_per, int) and effective_n_per >= 1

    input_ids = sorted(dataset.sanitized_varying_inputs.keys())
    existing_responses = result.pre_filtered_baseline_responses_by_input
    existing_acceptances = result.pre_filtered_baseline_acceptances_by_input
    changed = False

    if existing_responses is not None or existing_acceptances is not None:
        if existing_responses is None or existing_acceptances is None:
            raise ValueError(
                "pre_filtered_baseline_responses_by_input and "
                "pre_filtered_baseline_acceptances_by_input must both be present "
                "or both be absent"
            )
        if set(existing_responses.keys()) != set(input_ids):
            raise ValueError(
                "pre-filter baseline responses keys do not match sanitized inputs"
            )
        if set(existing_acceptances.keys()) != set(input_ids):
            raise ValueError(
                "pre-filter baseline acceptances keys do not match sanitized inputs"
            )

        for input_id in input_ids:
            responses_for_input = existing_responses.get(input_id)
            acceptances_for_input = existing_acceptances.get(input_id)
            if responses_for_input is None or acceptances_for_input is None:
                raise ValueError(
                    f"Missing pre-filter baseline data for input {input_id}"
                )
            if len(responses_for_input) != effective_n_per:
                raise ValueError(
                    f"Input {input_id}: expected {effective_n_per} pre-filter responses, "
                    f"found {len(responses_for_input)}"
                )
            if set(responses_for_input.keys()) != set(acceptances_for_input.keys()):
                raise ValueError(
                    f"Input {input_id}: pre-filter response ids do not match "
                    "acceptance ids"
                )
    else:
        expanded_params_list: list[dict[str, str]] = []
        for idx in input_ids:
            ip = dataset.get_input_params(idx)
            expanded_params_list.extend([ip] * effective_n_per)

        responses = responses_generator.generate(
            dataset.input_template,
            expanded_params_list,
        )

        expected_total = len(input_ids) * effective_n_per
        if len(responses) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} pre-filter baseline responses, "
                f"received {len(responses)}"
            )

        new_responses: dict[InputId, dict[ResponseId, str]] = {}
        new_acceptances: dict[InputId, dict[ResponseId, int | None]] = {}

        cursor = 0
        for idx in input_ids:
            per_input_resps = responses[cursor : cursor + effective_n_per]
            if len(per_input_resps) != effective_n_per:
                raise RuntimeError(
                    f"Input {idx}: expected {effective_n_per} responses, received "
                    f"{len(per_input_resps)}"
                )
            cursor += effective_n_per

            new_ids: list[ResponseId] = [ResponseId() for _ in range(effective_n_per)]
            new_responses[idx] = dict(zip(new_ids, per_input_resps, strict=True))

            accept_list = bias_tester.calculate_acceptances_for_responses(
                idx,
                per_input_resps,
            )
            if len(accept_list) != effective_n_per:
                raise RuntimeError(
                    f"Input {idx}: expected {effective_n_per} acceptance values, received "
                    f"{len(accept_list)}"
                )
            new_acceptances[idx] = dict(zip(new_ids, accept_list, strict=True))

        result.pre_filtered_baseline_responses_by_input = new_responses
        result.pre_filtered_baseline_acceptances_by_input = new_acceptances
        changed = True

    if result.filtered_varying_inputs is None:
        if input_prefilter is None:
            result.filtered_varying_inputs = input_ids
        else:
            print(f"Filtering {len(input_ids)} inputs...")
            assert result.pre_filtered_baseline_acceptances_by_input is not None, (
                "Pre-filter acceptances must be present before filtering"
            )
            result.filtered_varying_inputs = input_prefilter.filter_inputs(
                input_ids,
                result.pre_filtered_baseline_acceptances_by_input,
            )
            print(
                f"Filtered inputs from {len(input_ids)} to "
                f"{len(result.filtered_varying_inputs)}"
            )
        changed = True

    if changed:
        save_result(result, get_result_path(result.experiment_key, output_dir))


def collect_baseline_responses_by_input_if_needed(
    result: ConceptPipelineResult,
    *,
    dataset: ConceptPipelineDataset,
    responses_generator: ResponsesGenerator,
    input_prefilter: InputPrefilter | None,
    output_dir: Path,
    n_baseline_responses_pre_filter_per_input: int,
    n_baseline_responses_per_input: int,
) -> None:
    """Derive or validate the baseline pool (post-filter) from pre-filter data."""
    assert result.pre_filtered_baseline_responses_by_input is not None
    assert result.pre_filtered_baseline_acceptances_by_input is not None
    assert dataset.sanitized_varying_inputs is not None

    expected_per_input = n_baseline_responses_per_input
    assert isinstance(expected_per_input, int) and expected_per_input >= 1

    baseline_responses = result.baseline_responses_by_input
    baseline_acceptances = result.baseline_acceptances_by_input
    input_ids = list(dataset.sanitized_varying_inputs.keys())

    if baseline_responses is not None or baseline_acceptances is not None:
        if baseline_responses is None or baseline_acceptances is None:
            raise ValueError(
                "baseline_responses_by_input and baseline_acceptances_by_input "
                "must both be present or both be absent"
            )

        if set(baseline_responses.keys()) != set(input_ids):
            raise ValueError(
                "baseline_responses_by_input keys do not match sanitized inputs"
            )
        if set(baseline_acceptances.keys()) != set(input_ids):
            raise ValueError(
                "baseline_acceptances_by_input keys do not match sanitized inputs"
            )

        for input_id in input_ids:
            responses_for_input = baseline_responses.get(input_id)
            acceptances_for_input = baseline_acceptances.get(input_id)
            if responses_for_input is None or acceptances_for_input is None:
                raise ValueError(
                    f"Missing baseline responses/acceptances for input {input_id}"
                )
            if len(responses_for_input) != expected_per_input:
                raise ValueError(
                    f"Input {input_id}: expected {expected_per_input} baseline "
                    f"responses, found {len(responses_for_input)}"
                )
            if set(responses_for_input.keys()) != set(acceptances_for_input.keys()):
                raise ValueError(
                    f"Input {input_id}: baseline response ids do not match "
                    "acceptance ids"
                )

        responses_generator.export_baseline_responses_html(dataset, result, output_dir)
        return

    pre_filter_expected = (
        n_baseline_responses_pre_filter_per_input
        if input_prefilter is not None
        else n_baseline_responses_per_input
    )
    assert isinstance(pre_filter_expected, int)
    assert pre_filter_expected >= expected_per_input

    new_baseline_responses: dict[InputId, dict[ResponseId, str]] = {}
    new_baseline_acceptances: dict[InputId, dict[ResponseId, int | None]] = {}

    for input_id in input_ids:
        pre_responses = result.pre_filtered_baseline_responses_by_input.get(input_id)
        pre_acceptances = result.pre_filtered_baseline_acceptances_by_input.get(
            input_id
        )
        if pre_responses is None or pre_acceptances is None:
            raise ValueError(f"Missing pre-filter baseline data for input {input_id}")
        if len(pre_responses) != pre_filter_expected:
            raise ValueError(
                f"Input {input_id}: expected {pre_filter_expected} pre-filter responses, "
                f"found {len(pre_responses)}"
            )
        if set(pre_responses.keys()) != set(pre_acceptances.keys()):
            raise ValueError(
                f"Input {input_id}: pre-filter response ids do not match acceptance ids"
            )

        response_ids = sorted(pre_responses.keys())
        if len(response_ids) < expected_per_input:
            raise ValueError(
                f"Input {input_id}: expected at least {expected_per_input} pre-filter responses to select from"
            )
        response_ids = response_ids[:expected_per_input]

        selected_responses = {rid: pre_responses[rid] for rid in response_ids}
        selected_acceptances = {rid: pre_acceptances[rid] for rid in response_ids}

        if len(selected_responses) != expected_per_input:
            raise RuntimeError(
                f"Input {input_id}: failed to select {expected_per_input} baseline "
                "responses"
            )

        new_baseline_responses[input_id] = selected_responses
        new_baseline_acceptances[input_id] = selected_acceptances

    result.baseline_responses_by_input = new_baseline_responses
    result.baseline_acceptances_by_input = new_baseline_acceptances

    responses_generator.export_baseline_responses_html(dataset, result, output_dir)
    save_result(result, get_result_path(result.experiment_key, output_dir))
