from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
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
from biases_in_the_blind_spot.concept_pipeline.plotting import plot_bias_impact
from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)


def _compute_flipped_variation_pairs(
    bias_result,
) -> dict[InputId, dict[VariationPairId, bool]]:
    """Compute input_id -> pair_id -> whether acceptances flip between positive and negative.

    Assumption (enforced): exactly one positive and one negative response per pair.
    """
    assert hasattr(bias_result, "responses_by_input")
    responses_by_input = bias_result.responses_by_input
    assert isinstance(responses_by_input, dict)

    out: dict[InputId, dict[VariationPairId, bool]] = {}
    for input_id, by_pair in responses_by_input.items():
        assert isinstance(by_pair, dict) and len(by_pair) > 0
        per_input: dict[VariationPairId, bool] = {}
        for pair_id, pair_responses in by_pair.items():
            assert isinstance(pair_responses, VariationPairResponses)

            # Fail-fast: "flip" semantics are only clear when each side has exactly one response.
            assert len(pair_responses.positive_responses) == 1, (
                "Expected exactly 1 positive response per variation pair. "
                f"Got {len(pair_responses.positive_responses)} for input={input_id}, pair={pair_id}."
            )
            assert len(pair_responses.negative_responses) == 1, (
                "Expected exactly 1 negative response per variation pair. "
                f"Got {len(pair_responses.negative_responses)} for input={input_id}, pair={pair_id}."
            )
            assert len(pair_responses.positive_acceptances) == 1
            assert len(pair_responses.negative_acceptances) == 1
            assert set(pair_responses.positive_responses.keys()) == set(
                pair_responses.positive_acceptances.keys()
            )
            assert set(pair_responses.negative_responses.keys()) == set(
                pair_responses.negative_acceptances.keys()
            )

            per_input[pair_id] = pair_responses.has_flipped_acceptance()
        out[input_id] = per_input
    assert len(out) > 0
    return out


def test_variations_bias_for_stage(
    result: ConceptPipelineResult,
    stage: StageResults,
    *,
    dataset: ConceptPipelineDataset,
    bias_tester: BiasTester,
    responses_generator: ResponsesGenerator,
    output_dir: Path,
) -> None:
    assert result.stages is not None and len(result.stages) > 0
    current_stage = stage
    assert dataset.variations is not None

    prev_stage = (
        result.stages[current_stage.stage_idx - 1]
        if current_stage.stage_idx > 0
        else None
    )

    field_missing = current_stage.variation_bias_results is None
    if field_missing:
        current_stage.variation_bias_results = {}

    result_path = get_result_path(result.experiment_key, output_dir)
    # Only process concepts that passed baseline verbalization (unverbalized on baseline).
    # Concepts that failed baseline verbalization should not be evaluated for variation bias.
    assert current_stage.concept_ids_unverbalized_on_baseline is not None, (
        "concept_ids_unverbalized_on_baseline must be set before testing variation bias"
    )
    unverbalized_concepts_set = set(current_stage.concept_ids_unverbalized_on_baseline)
    stage_inputs = set(current_stage.get_stage_input_ids())

    expected_variations: dict[
        ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]
    ] = {}
    for concept_id in sorted(dataset.variations.keys()):
        if concept_id not in unverbalized_concepts_set:
            continue
        concept_variations = dataset.variations.get(concept_id, {})
        assert isinstance(concept_variations, dict) and len(concept_variations) > 0
        stage_concept_variations = {
            k: v for k, v in concept_variations.items() if k in stage_inputs
        }
        if len(stage_concept_variations) == 0:
            raise ValueError(
                f"No variations for concept {concept_id} in current stage inputs"
            )
        expected_variations[concept_id] = stage_concept_variations

    # If this is a fresh stage, reuse variation-bias responses from the previous stage
    # for overlapping inputs/concepts. This avoids regenerating responses (and thus new
    # response ids) for the same (concept,input,pair) items, which would break downstream
    # carry-over of variation verbalization keyed by response ids.
    if (
        field_missing
        and prev_stage is not None
        and prev_stage.variation_bias_results is not None
    ):
        assert current_stage.variation_bias_results is not None
        for concept_id, stage_concept_variations in expected_variations.items():
            prev_res = prev_stage.variation_bias_results.get(concept_id)
            if prev_res is None:
                continue
            expected_inputs = set(stage_concept_variations.keys())
            prev_inputs = set(prev_res.responses_by_input.keys())
            shared_inputs = expected_inputs & prev_inputs
            if not shared_inputs:
                continue

            # Copy only the shared inputs; we keep the same VariationPairResponses objects
            # so their ResponseId keys remain stable across stages.
            carried_responses_by_input = {
                input_idx: prev_res.responses_by_input[input_idx]
                for input_idx in shared_inputs
            }
            current_stage.variation_bias_results[concept_id] = type(prev_res)(
                responses_by_input=carried_responses_by_input,
                statistics_positive_vs_negative=prev_res.statistics_positive_vs_negative,
                statistics_positive=prev_res.statistics_positive,
                statistics_negative=prev_res.statistics_negative,
                # Do NOT carry over flipped_variation_pairs across stages.
                # Stage-specific results will include a different set of inputs/pairs (that are carried over from the previous stage plus the new ones), and we
                # recompute this field from responses_by_input once the stage is complete.
                flipped_variation_pairs=None,
            )

    if not field_missing:
        assert current_stage.variation_bias_results is not None
        existing_keys = set(current_stage.variation_bias_results.keys())
        expected_keys = set(expected_variations.keys())
        extra = existing_keys - expected_keys
        missing = expected_keys - existing_keys
        if extra:
            raise ValueError(
                f"Variation bias results contain unexpected concepts: {sorted(extra)}"
            )
        if missing:
            raise ValueError(
                f"Variation bias results missing concepts: {sorted(missing)}"
            )

        for concept_id, stage_concept_variations in expected_variations.items():
            res = current_stage.variation_bias_results.get(concept_id)
            if res is None:
                raise ValueError(f"Missing variation bias result for {concept_id}")
            expected_inputs = set(stage_concept_variations.keys())
            actual_inputs = set(res.responses_by_input.keys())
            if actual_inputs != expected_inputs:
                raise ValueError(
                    f"Variation bias responses_by_input mismatch for concept {concept_id}: "
                    f"expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}"
                )
            expected_pairs = sum(len(v) for v in stage_concept_variations.values())
            if hasattr(res, "num_variations"):
                if res.num_variations != expected_pairs:
                    raise ValueError(
                        f"Variation bias num_variations mismatch for concept {concept_id}: "
                        f"expected {expected_pairs}, got {res.num_variations}"
                    )
            for input_idx, pair_map in stage_concept_variations.items():
                actual_pair_map = res.responses_by_input.get(input_idx, {})
                if set(actual_pair_map.keys()) != set(pair_map.keys()):
                    raise ValueError(
                        f"Variation pairs mismatch for concept {concept_id}, input {input_idx}"
                    )

            computed_flips = _compute_flipped_variation_pairs(res)
            existing_flips = getattr(res, "flipped_variation_pairs", None)
            if existing_flips is None:
                res.flipped_variation_pairs = computed_flips
            elif existing_flips != computed_flips:
                raise ValueError(
                    "Existing flipped_variation_pairs does not match freshly computed value "
                    f"for concept {concept_id}"
                )
        return

    # Generate only the missing inputs (anything carried over above should be reused).
    assert current_stage.variation_bias_results is not None
    missing_variations: dict[
        ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]
    ] = {}
    for concept_id, stage_concept_variations in expected_variations.items():
        existing = current_stage.variation_bias_results.get(concept_id)
        existing_inputs = set(existing.responses_by_input.keys()) if existing else set()
        expected_inputs = set(stage_concept_variations.keys())
        missing_inputs = expected_inputs - existing_inputs
        if missing_inputs:
            missing_variations[concept_id] = {
                input_idx: stage_concept_variations[input_idx]
                for input_idx in missing_inputs
            }

    concept_ids_to_run = list(expected_variations.keys())
    assert current_stage.variation_bias_results is not None
    if concept_ids_to_run:
        # If everything was carried over, we can validate and finish without any generation.
        if len(missing_variations) == 0:
            for res in current_stage.variation_bias_results.values():
                computed_flips = _compute_flipped_variation_pairs(res)
                if getattr(res, "flipped_variation_pairs", None) is None:
                    res.flipped_variation_pairs = computed_flips
                else:
                    assert res.flipped_variation_pairs == computed_flips

            save_result(result, result_path)
            return

        assert bias_tester is not None
        assert responses_generator is not None
        input_template = dataset.input_template
        assert isinstance(input_template, str)

        def _validate_variation_bias_complete() -> None:
            missing_counts = 0
            assert current_stage.variation_bias_results is not None
            for concept_id in concept_ids_to_run:
                per_input_results = current_stage.variation_bias_results.get(concept_id)
                if per_input_results is None:
                    raise ValueError(
                        f"variation_bias_results missing concept {concept_id}"
                    )
                for (
                    input_idx,
                    variation_pairs_for_input,
                ) in expected_variations[concept_id].items():
                    if input_idx not in per_input_results.responses_by_input:
                        raise ValueError(
                            f"variation_bias_results missing input {input_idx} for concept {concept_id}"
                        )
                    variation_pairs_found = per_input_results.responses_by_input[
                        input_idx
                    ]
                    if set(variation_pairs_found.keys()) != set(
                        variation_pairs_for_input.keys()
                    ):
                        missing_counts += 1
                        print(
                            f"Variation pairs mismatch for concept {concept_id}, input {input_idx}."
                        )
            if missing_counts > 0:
                raise ValueError(
                    "variation_bias_results is missing expected inputs or pairs after bias test"
                )

        missing_inputs_all = {
            input_idx
            for by_input in missing_variations.values()
            for input_idx in by_input.keys()
        }
        input_params_by_input = {
            input_idx: dataset.get_input_params(input_idx)
            for input_idx in missing_inputs_all
        }

        per_concept_results = bias_tester.test_variatons_batch(
            input_template=input_template,
            input_parameters_by_input=input_params_by_input,
            variations_by_concept=missing_variations,
        )

        for concept_id, generated in per_concept_results.items():
            dest = current_stage.variation_bias_results.get(concept_id)
            if dest is None:
                current_stage.variation_bias_results[concept_id] = generated
            else:
                for inp, pairs in generated.responses_by_input.items():
                    if inp in dest.responses_by_input:
                        raise ValueError(
                            f"Duplicate variation bias input {inp} for concept {concept_id}"
                        )
                    dest.responses_by_input[inp] = pairs

        _validate_variation_bias_complete()
        for concept_id in current_stage.variation_bias_results.keys():
            bias_tester.recompute_statistics(
                current_stage.variation_bias_results[concept_id]
            )
            res = current_stage.variation_bias_results[concept_id]
            computed_flips = _compute_flipped_variation_pairs(res)
            if getattr(res, "flipped_variation_pairs", None) is None:
                res.flipped_variation_pairs = computed_flips
            else:
                assert res.flipped_variation_pairs == computed_flips

        save_result(result, result_path)

        model_name = responses_generator.model_name
        plot_bias_impact(
            dataset,
            result,
            result.get_stage_figures_root(output_dir, current_stage),
            current_stage,
            model_name=model_name,
        )

        assert current_stage.variation_bias_results is not None
        concept_ids_with_bias_results: list[ConceptId] = [
            cid
            for cid in current_stage.variation_bias_results.keys()
            if cid in unverbalized_concepts_set
        ]

        use_mcnemar = result.significance_test == "mcnemar"
        print(f"Concepts with bias results: {len(concept_ids_with_bias_results)}")

        save_result(result, result_path)

        assert current_stage.variation_bias_results is not None
        lines: list[str] = []
        for concept_id in concept_ids_with_bias_results:
            res = current_stage.variation_bias_results.get(concept_id)
            if res is None:
                continue
            stats = res.statistics_positive_vs_negative
            if isinstance(stats, dict):
                diff = stats.get("proportion_difference")
                pval_key = "mcnemar_p_value" if use_mcnemar else "fisher_p_value"
                pval = stats.get(pval_key)
                pos_rate = stats.get("positive_proportion")
                neg_rate = stats.get("negative_proportion")
                title = dataset.get_concept_title(concept_id)
                assert isinstance(pos_rate, float) and isinstance(neg_rate, float)
                assert isinstance(pval, float) and isinstance(diff, float)
                lines.append(
                    f" - {title}: diff={diff:.3f}, positive={pos_rate:.3f}, negative={neg_rate:.3f}, p={pval:.3g}"
                )

        if len(lines) > 0:
            print(f"Details for {len(lines)} concepts:")
            for line in lines:
                print(line)
        else:
            print("No concepts to report")
