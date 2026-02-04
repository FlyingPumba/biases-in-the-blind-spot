from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.pipeline_persistence import (
    get_result_path,
    save_result,
)
from biases_in_the_blind_spot.concept_pipeline.plotting import (
    plot_concept_baseline_verbalization,
)
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)


async def analyze_verbalization_on_baseline_for_stage(
    result: ConceptPipelineResult,
    stage: StageResults,
    *,
    dataset: ConceptPipelineDataset,
    output_dir: Path,
    baseline_verbalization_threshold: float,
    verbalization_detector: VerbalizationDetector,
) -> None:
    """Stage-aware baseline verbalization analysis."""
    assert result.baseline_responses_by_input is not None
    assert result.stages is not None and len(result.stages) > 0
    if (
        result.baseline_verbalization_threshold is not None
        and result.baseline_verbalization_threshold != baseline_verbalization_threshold
    ):
        raise ValueError(
            "Baseline verbalization threshold mismatch between stored result and argument"
        )

    current_stage = stage
    # Optimization: baseline verbalization check is only run at stage 0.
    # For stage>0, we treat all stage-start concepts as "unverbalized on baseline"
    # and do not store baseline verbalization flags for the stage.
    if current_stage.stage_idx > 0:
        expected_unverbalized = list(current_stage.concepts_at_stage_start)
        if current_stage.concept_verbalization_on_baseline_responses is not None:
            raise ValueError(
                "concept_verbalization_on_baseline_responses must be None for stage>0 "
                f"(stage_idx={current_stage.stage_idx})"
            )
        existing = current_stage.concept_ids_unverbalized_on_baseline
        if existing is None:
            current_stage.concept_ids_unverbalized_on_baseline = expected_unverbalized
        else:
            if list(existing) != expected_unverbalized:
                raise ValueError(
                    "Existing concept_ids_unverbalized_on_baseline does not match expected "
                    f"for stage {current_stage.stage_idx}: "
                    f"stored={len(existing)}, expected={len(expected_unverbalized)}"
                )
        save_result(result, get_result_path(result.experiment_key, output_dir))
        return

    prev_stage = (
        result.stages[current_stage.stage_idx - 1]
        if current_stage.stage_idx > 0
        else None
    )

    current_inputs: set[InputId] = set(current_stage.get_stage_input_ids())
    prev_inputs: set[InputId] = (
        set(prev_stage.get_stage_input_ids()) if prev_stage is not None else set()
    )
    new_inputs: set[InputId] = current_inputs - prev_inputs
    assert len(current_inputs) > 0
    assert set(current_inputs).issubset(result.baseline_responses_by_input.keys())

    def _compute_unverbalized_from_flags(
        concept_ids: list[ConceptId],
        flags_map: dict[
            ConceptId, dict[InputId, dict[ResponseId, VerbalizationCheckResult]]
        ],
    ) -> list[ConceptId]:
        assert result.baseline_verbalization_threshold is not None
        thr = float(result.baseline_verbalization_threshold)
        unverbalized: list[ConceptId] = []
        for cid in concept_ids:
            per_concept_flags = flags_map.get(cid, {})
            all_flags: list[bool] = []
            for flags_map_per_input in per_concept_flags.values():
                assert (
                    isinstance(flags_map_per_input, dict)
                    and len(flags_map_per_input) > 0
                )
                all_flags.extend(
                    result_obj.verbalized for result_obj in flags_map_per_input.values()
                )
            if len(all_flags) == 0:
                raise ValueError(
                    f"No baseline verbalization flags available for concept {cid} in current stage"
                )
            positives = sum(1 for v in all_flags if v)
            ratio = positives / len(all_flags)
            if ratio < thr:
                unverbalized.append(cid)
        return unverbalized

    if current_stage.concept_verbalization_on_baseline_responses is None:
        current_stage.concept_verbalization_on_baseline_responses = {}
    concept_ids_in_stage: list[ConceptId] = current_stage.concepts_at_stage_start

    if current_stage.concept_verbalization_on_baseline_responses:
        concept_keys = set(
            current_stage.concept_verbalization_on_baseline_responses.keys()
        )
        expected_concepts = set(concept_ids_in_stage)
        extra = concept_keys - expected_concepts
        missing = expected_concepts - concept_keys
        if extra:
            raise ValueError(
                f"Baseline verbalization data has unexpected concepts: {sorted(extra)}"
            )
        if missing:
            raise ValueError(
                f"Baseline verbalization data missing concepts: {sorted(missing)}"
            )

        for cid in concept_ids_in_stage:
            by_input = current_stage.concept_verbalization_on_baseline_responses.get(
                cid, {}
            )
            input_keys = set(by_input.keys())
            extra_inputs = input_keys - current_inputs
            missing_inputs = current_inputs - input_keys
            if extra_inputs:
                raise ValueError(
                    f"Baseline verbalization data for concept {cid} has unexpected inputs: {sorted(extra_inputs)}"
                )
            if missing_inputs:
                raise ValueError(
                    f"Baseline verbalization data for concept {cid} is missing inputs: {sorted(missing_inputs)}"
                )
            for input_idx in current_inputs:
                flags_map = by_input.get(input_idx, {})
                resp_map = result.baseline_responses_by_input[input_idx]
                if set(flags_map.keys()) != set(resp_map.keys()):
                    raise ValueError(
                        f"Baseline verbalization response ids mismatch for concept {cid}, input {input_idx}"
                    )

        # Immutable path: compute ratios, validate stored unverbalized list, and exit
        by_concept = current_stage.concept_verbalization_on_baseline_responses
        print(f"Baseline verbalization ratios for stage {current_stage.stage_idx}")
        expected_unverbalized = _compute_unverbalized_from_flags(
            concept_ids_in_stage,
            by_concept,
        )
        for cid in concept_ids_in_stage:
            flags_by_input = by_concept.get(cid, {})
            flags_flat: list[bool] = []
            for flags_map in flags_by_input.values():
                assert isinstance(flags_map, dict) and len(flags_map) > 0
                flags_flat.extend(
                    result_obj.verbalized for result_obj in flags_map.values()
                )
            assert len(flags_flat) > 0, (
                f"No baseline verbalization flags available for concept {cid} in current stage"
            )
            positives = sum(1 for v in flags_flat if v)
            ratio = positives / len(flags_flat)
            print(
                f" - {cid} - {dataset.get_concept_title(cid)}: {ratio:.2f} ({positives}/{len(flags_flat)})"
                + (" -> UNVERBALIZED" if cid in expected_unverbalized else "")
            )

        if current_stage.concept_ids_unverbalized_on_baseline is None:
            raise ValueError(
                "Baseline verbalization data present but concept_ids_unverbalized_on_baseline missing"
            )
        if current_stage.concept_ids_unverbalized_on_baseline != expected_unverbalized:
            raise ValueError(
                "Existing concept_ids_unverbalized_on_baseline does not match computed values"
            )

        assert result.baseline_verbalization_threshold is not None
        thr = float(result.baseline_verbalization_threshold)
        print(
            f"There are {len(expected_unverbalized)} concepts out of {len(concept_ids_in_stage)} below the baseline verbalization threshold ({thr:.2f})"
        )
        save_result(result, get_result_path(result.experiment_key, output_dir))
        plot_concept_baseline_verbalization(
            dataset,
            result,
            result.get_stage_figures_root(output_dir, current_stage),
            current_stage,
        )
        return

    if prev_stage is not None and (
        prev_stage.concept_verbalization_on_baseline_responses is not None
    ):
        for cid in concept_ids_in_stage:
            prev_map = prev_stage.concept_verbalization_on_baseline_responses.get(cid)
            if prev_map is None:
                continue
            if cid not in current_stage.concept_verbalization_on_baseline_responses:
                current_stage.concept_verbalization_on_baseline_responses[cid] = {
                    input_idx: {
                        resp_id: VerbalizationCheckResult(
                            verbalized=result_obj.verbalized,
                            witness=result_obj.witness,
                        )
                        for resp_id, result_obj in flags.items()
                    }
                    for input_idx, flags in prev_map.items()
                }
            else:
                existing_by_input = (
                    current_stage.concept_verbalization_on_baseline_responses[cid]
                )
                for input_idx, flags in prev_map.items():
                    if input_idx not in existing_by_input:
                        raise ValueError(
                            f"Baseline verbalization for concept {cid} missing input {input_idx} that exists in previous stage"
                        )
                    existing_flags = existing_by_input.get(input_idx, {})
                    for resp_id, result_obj in flags.items():
                        if resp_id not in existing_flags:
                            raise ValueError(
                                f"Baseline verbalization for concept {cid}, input {input_idx} missing response id {resp_id} that exists in previous stage"
                            )
                        existing_result = existing_flags[resp_id]
                        if (
                            existing_result.verbalized != result_obj.verbalized
                            or existing_result.witness != result_obj.witness
                        ):
                            raise ValueError(
                                f"Baseline verbalization mismatch for concept {cid}, input {input_idx}, response {resp_id}"
                            )

    requests_by_concept: dict[ConceptId, dict[InputId, dict[ResponseId, str]]] = {}
    for cid in concept_ids_in_stage:
        flags_by_input = current_stage.concept_verbalization_on_baseline_responses.get(
            cid, {}
        )
        missing_inputs = new_inputs - set(flags_by_input.keys())
        if not missing_inputs:
            continue
        per_concept_requests: dict[InputId, dict[ResponseId, str]] = {}
        for input_idx in sorted(missing_inputs):
            responses = result.baseline_responses_by_input[input_idx]
            if len(responses) == 0:
                raise ValueError(
                    f"No baseline responses for input {input_idx} to analyze verbalization"
                )
            per_concept_requests[input_idx] = dict(responses)
        requests_by_concept[cid] = per_concept_requests

    if requests_by_concept:
        analysis_results = await verbalization_detector.is_verbalized_baseline_batch(
            dataset, requests_by_concept
        )
        if set(analysis_results.keys()) != set(requests_by_concept.keys()):
            raise ValueError(
                "Baseline verbalization analysis returned mismatched concept ids"
            )

        for concept_id, per_input_results in analysis_results.items():
            flags_by_input = (
                current_stage.concept_verbalization_on_baseline_responses.setdefault(
                    concept_id, {}
                )
            )
            expected_inputs = set(requests_by_concept[concept_id].keys())
            if set(per_input_results.keys()) != expected_inputs:
                raise ValueError(
                    f"Baseline verbalization results for concept {concept_id} missing or extra inputs"
                )
            for input_idx, result_map in per_input_results.items():
                if input_idx in flags_by_input:
                    raise ValueError(
                        f"Baseline verbalization for concept {concept_id} input {input_idx} already set"
                    )
                expected_resp_ids = set(
                    result.baseline_responses_by_input[input_idx].keys()
                )
                if set(result_map.keys()) != expected_resp_ids:
                    raise ValueError(
                        f"Baseline verbalization response ids mismatch for concept {concept_id}, input {input_idx}"
                    )
                flags_by_input[input_idx] = dict(result_map)

    concept_ids_in_stage = current_stage.concepts_at_stage_start
    assert current_stage.concept_verbalization_on_baseline_responses is not None
    by_concept = current_stage.concept_verbalization_on_baseline_responses
    print(f"Baseline verbalization ratios for stage {current_stage.stage_idx}")
    expected_unverbalized = _compute_unverbalized_from_flags(
        concept_ids_in_stage,
        by_concept,
    )
    for cid in concept_ids_in_stage:
        flags_by_input = by_concept.get(cid, {})
        all_flags: list[bool] = []
        for flags_map in flags_by_input.values():
            assert isinstance(flags_map, dict) and len(flags_map) > 0
            all_flags.extend(result_obj.verbalized for result_obj in flags_map.values())
        assert len(all_flags) > 0, (
            f"No baseline verbalization flags available for concept {cid} in current stage"
        )
        positives = sum(1 for v in all_flags if v)
        ratio = positives / len(all_flags)
        print(
            f" - {cid} - {dataset.get_concept_title(cid)}: {ratio:.2f} ({positives}/{len(all_flags)})"
            + (" -> UNVERBALIZED" if cid in expected_unverbalized else "")
        )

    if current_stage.concept_ids_unverbalized_on_baseline is None:
        current_stage.concept_ids_unverbalized_on_baseline = expected_unverbalized
    else:
        if current_stage.concept_ids_unverbalized_on_baseline != expected_unverbalized:
            raise ValueError(
                "Existing concept_ids_unverbalized_on_baseline does not match computed values"
            )

    assert result.baseline_verbalization_threshold is not None
    thr = float(result.baseline_verbalization_threshold)
    print(
        f"There are {len(expected_unverbalized)} concepts out of {len(concept_ids_in_stage)} below the baseline verbalization threshold ({thr:.2f})"
    )
    save_result(result, get_result_path(result.experiment_key, output_dir))
    plot_concept_baseline_verbalization(
        dataset,
        result,
        result.get_stage_figures_root(output_dir, current_stage),
        current_stage,
    )
