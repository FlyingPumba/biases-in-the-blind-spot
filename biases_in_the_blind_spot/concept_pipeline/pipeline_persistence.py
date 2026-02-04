import os
from pathlib import Path
from typing import Literal

from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)


def get_result_path(experiment_key: str, output_dir: Path) -> Path:
    return output_dir / f"concept_pipeline_{experiment_key}.json"


def save_result(result: ConceptPipelineResult, result_path: Path) -> None:
    """Persist the full pipeline result plus per-stage files atomically."""
    if result.stages is not None and len(result.stages) > 0:
        base_name = result_path.stem
        parent_dir = result_path.parent

        for stage in result.stages:
            stage_filename = f"{base_name}_stage-{stage.stage_idx}.json"
            stage_path = parent_dir / stage_filename
            stage_data = stage.to_json(indent=4, ensure_ascii=False)
            tmp_path = stage_path.with_suffix(stage_path.suffix + ".tmp")
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(stage_data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, stage_path)

    original_stages = result.stages
    result.stages = None
    data = result.to_json(indent=4, ensure_ascii=False)
    result.stages = original_stages

    tmp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)


def populate_configs_if_needed(
    result: ConceptPipelineResult,
    *,
    responses_generator_config,
    verbalization_detector_config,
    bias_tester_config,
    significance_test: Literal["fisher", "mcnemar"],
    n_baseline_responses_pre_filter_per_input: int,
    n_baseline_responses_per_input: int,
    input_prefilter_is_present: bool,
    parsed_labels_mapping: dict[int, str] | None,
    variations_verbalization_threshold: float,
    baseline_verbalization_threshold: float,
    futility_stop_power_threshold: float,
    significant_concepts_p_value_alpha: float,
    apply_bonferroni_correction: bool,
) -> None:
    """Fill missing config/threshold fields in the result from pipeline defaults."""
    expected_pre_filter_responses = (
        n_baseline_responses_pre_filter_per_input
        if input_prefilter_is_present
        else n_baseline_responses_per_input
    )
    if result.responses_generator_config is None:
        result.responses_generator_config = responses_generator_config
    else:
        if result.responses_generator_config != responses_generator_config:
            raise ValueError(
                "responses_generator_config mismatch: "
                f"stored={result.responses_generator_config}, "
                f"pipeline={responses_generator_config}"
            )
    if result.verbalization_detector_config is None:
        result.verbalization_detector_config = verbalization_detector_config
    else:
        if result.verbalization_detector_config != verbalization_detector_config:
            raise ValueError(
                "verbalization_detector_config mismatch: "
                f"stored={result.verbalization_detector_config}, "
                f"pipeline={verbalization_detector_config}"
            )
    if result.bias_tester_config is None:
        result.bias_tester_config = bias_tester_config
    else:
        if result.bias_tester_config != bias_tester_config:
            raise ValueError(
                "bias_tester_config mismatch: "
                f"stored={result.bias_tester_config}, "
                f"pipeline={bias_tester_config}"
            )
    if result.significance_test is None:
        result.significance_test = significance_test
    else:
        if result.significance_test != significance_test:
            raise ValueError(
                "significance_test mismatch: "
                f"stored={result.significance_test}, "
                f"pipeline={significance_test}"
            )
    if result.n_baseline_responses_pre_filter_per_input is None:
        result.n_baseline_responses_pre_filter_per_input = expected_pre_filter_responses
    else:
        if (
            result.n_baseline_responses_pre_filter_per_input
            != expected_pre_filter_responses
        ):
            raise ValueError(
                "n_baseline_responses_pre_filter_per_input mismatch: "
                f"stored={result.n_baseline_responses_pre_filter_per_input}, "
                f"pipeline_expected={expected_pre_filter_responses}"
            )
    if result.n_baseline_responses_per_input is None:
        result.n_baseline_responses_per_input = n_baseline_responses_per_input
    else:
        if result.n_baseline_responses_per_input != n_baseline_responses_per_input:
            raise ValueError(
                "n_baseline_responses_per_input mismatch: "
                f"stored={result.n_baseline_responses_per_input}, "
                f"pipeline={n_baseline_responses_per_input}"
            )
    if result.parsed_labels_mapping is None:
        result.parsed_labels_mapping = parsed_labels_mapping
    else:
        if result.parsed_labels_mapping != parsed_labels_mapping:
            raise ValueError(
                "parsed_labels_mapping mismatch: "
                f"stored={result.parsed_labels_mapping}, "
                f"pipeline={parsed_labels_mapping}"
            )
    if result.variations_verbalization_threshold is None:
        result.variations_verbalization_threshold = variations_verbalization_threshold
    else:
        if (
            result.variations_verbalization_threshold
            != variations_verbalization_threshold
        ):
            raise ValueError(
                "variations_verbalization_threshold mismatch: "
                f"stored={result.variations_verbalization_threshold}, "
                f"pipeline={variations_verbalization_threshold}"
            )
    if result.baseline_verbalization_threshold is None:
        result.baseline_verbalization_threshold = baseline_verbalization_threshold
    else:
        if result.baseline_verbalization_threshold != baseline_verbalization_threshold:
            raise ValueError(
                "baseline_verbalization_threshold mismatch: "
                f"stored={result.baseline_verbalization_threshold}, "
                f"pipeline={baseline_verbalization_threshold}"
            )
    if result.futility_stop_power_threshold is None:
        result.futility_stop_power_threshold = futility_stop_power_threshold
    else:
        if result.futility_stop_power_threshold != futility_stop_power_threshold:
            raise ValueError(
                "futility_stop_power_threshold mismatch: "
                f"stored={result.futility_stop_power_threshold}, "
                f"pipeline={futility_stop_power_threshold}"
            )
    if result.significant_concepts_p_value_alpha is None:
        result.significant_concepts_p_value_alpha = significant_concepts_p_value_alpha
    else:
        if (
            result.significant_concepts_p_value_alpha
            != significant_concepts_p_value_alpha
        ):
            raise ValueError(
                "significant_concepts_p_value_alpha mismatch: "
                f"stored={result.significant_concepts_p_value_alpha}, "
                f"pipeline={significant_concepts_p_value_alpha}"
            )
    if result.apply_bonferroni_correction is None:
        result.apply_bonferroni_correction = apply_bonferroni_correction
    else:
        if result.apply_bonferroni_correction != apply_bonferroni_correction:
            raise ValueError(
                "apply_bonferroni_correction mismatch: "
                f"stored={result.apply_bonferroni_correction}, "
                f"pipeline={apply_bonferroni_correction}"
            )


def _load_result(result_path: Path) -> ConceptPipelineResult | None:
    if not result_path.exists():
        return None

    print(f"Loading result from {result_path.absolute()}")
    with open(result_path, encoding="utf-8") as f:
        result_data = f.read()
    print("Loaded result data")
    result = ConceptPipelineResult.from_json(result_data)
    print("Loaded result into object")

    base_name = result_path.stem
    parent_dir = result_path.parent
    stages = []
    stage_idx = 0

    while True:
        stage_filename = f"{base_name}_stage-{stage_idx}.json"
        stage_path = parent_dir / stage_filename
        if not stage_path.exists():
            break

        print(f"Loading stage {stage_idx} from {stage_path.name}")
        with open(stage_path, encoding="utf-8") as f:
            stage_data = f.read()
        stage = StageResults.from_json(stage_data)
        stages.append(stage)
        stage_idx += 1

    if len(stages) > 0:
        result.stages = stages  # type: ignore
        print(f"Loaded {len(stages)} stages from separate files")

    return result  # type: ignore


def load_pipeline_result_for_experiment(
    experiment_key: str, output_dir: Path
) -> ConceptPipelineResult | None:
    result_path = get_result_path(experiment_key, output_dir)
    return _load_result(result_path)
