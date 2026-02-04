from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)
from biases_in_the_blind_spot.concept_pipeline.pipeline_persistence import (
    get_result_path,
    populate_configs_if_needed,
)


def test_get_result_path():
    path = get_result_path("exp", Path("/tmp/out"))
    assert path.name == "concept_pipeline_exp.json"
    assert path.parent == Path("/tmp/out")


def test_populate_configs_sets_defaults_when_missing():
    result = ConceptPipelineResult()
    populate_configs_if_needed(
        result,
        responses_generator_config={"a": 1},
        verbalization_detector_config={"b": 2},
        bias_tester_config={"c": 3},
        significance_test="fisher",
        n_baseline_responses_pre_filter_per_input=5,
        n_baseline_responses_per_input=2,
        input_prefilter_is_present=True,
        parsed_labels_mapping={0: "N", 1: "Y"},
        variations_verbalization_threshold=0.4,
        baseline_verbalization_threshold=0.3,
        futility_stop_power_threshold=0.1,
        significant_concepts_p_value_alpha=0.05,
        apply_bonferroni_correction=True,
    )
    assert result.responses_generator_config == {"a": 1}
    assert result.verbalization_detector_config == {"b": 2}
    assert result.bias_tester_config == {"c": 3}
    assert result.significance_test == "fisher"
    assert result.n_baseline_responses_pre_filter_per_input == 5
    assert result.n_baseline_responses_per_input == 2
    assert result.parsed_labels_mapping == {0: "N", 1: "Y"}
    assert result.variations_verbalization_threshold == 0.4
    assert result.baseline_verbalization_threshold == 0.3
    assert result.futility_stop_power_threshold == 0.1
    assert result.significant_concepts_p_value_alpha == 0.05
    assert result.apply_bonferroni_correction is True


def test_populate_configs_raises_on_mismatch():
    result = ConceptPipelineResult(
        responses_generator_config={"a": 9},
        verbalization_detector_config={"b": 9},
        bias_tester_config={"c": 9},
        significance_test="mcnemar",
        n_baseline_responses_pre_filter_per_input=2,
        n_baseline_responses_per_input=2,
        parsed_labels_mapping={0: "N", 1: "Y"},
        variations_verbalization_threshold=0.4,
        baseline_verbalization_threshold=0.3,
        futility_stop_power_threshold=0.1,
        significant_concepts_p_value_alpha=0.05,
        apply_bonferroni_correction=True,
    )
    # responses_generator_config mismatch should raise
    try:
        populate_configs_if_needed(
            result,
            responses_generator_config={"a": 1},
            verbalization_detector_config={"b": 9},
            bias_tester_config={"c": 9},
            significance_test="mcnemar",
            n_baseline_responses_pre_filter_per_input=2,
            n_baseline_responses_per_input=2,
            input_prefilter_is_present=False,
            parsed_labels_mapping={0: "N", 1: "Y"},
            variations_verbalization_threshold=0.4,
            baseline_verbalization_threshold=0.3,
            futility_stop_power_threshold=0.1,
            significant_concepts_p_value_alpha=0.05,
            apply_bonferroni_correction=True,
        )
        assert False, "Expected mismatch to raise"  # noqa: B011
    except ValueError:
        pass
