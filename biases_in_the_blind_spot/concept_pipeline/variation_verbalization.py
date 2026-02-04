from pathlib import Path
from typing import cast

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
    plot_variation_verbalization,
)
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)


async def analyze_verbalization_on_variations_for_stage(
    result: ConceptPipelineResult,
    stage: StageResults,
    *,
    dataset: ConceptPipelineDataset,
    output_dir: Path,
    verbalization_detector: VerbalizationDetector,
) -> None:
    assert result.variations_verbalization_threshold is not None
    assert result.stages is not None and len(result.stages) > 0

    current_stage = stage
    prev_stage = (
        result.stages[current_stage.stage_idx - 1]
        if current_stage.stage_idx > 0
        else None
    )

    assert current_stage.variation_bias_results is not None
    assert current_stage.significant_concepts is not None

    futility_stopped = set(current_stage.futility_stopped_concepts or [])
    assert futility_stopped.isdisjoint(set(current_stage.significant_concepts)), (
        "Futility-stopped concepts must not be processed for variation verbalization; "
        f"overlap={sorted(futility_stopped & set(current_stage.significant_concepts))}"
    )

    current_inputs: set[InputId] = set(current_stage.get_stage_input_ids())
    if current_stage.concept_verbalization_on_variation_responses is None:
        current_stage.concept_verbalization_on_variation_responses = {}

    has_existing_data = bool(current_stage.concept_verbalization_on_variation_responses)

    def _expected_flipped_pairs_by_input(
        concept_id: ConceptId,
    ) -> dict[InputId, set[VariationPairId]]:
        """Return stage-scoped flipped pair ids, keyed by input id (only inputs in current stage)."""
        bias_map_local = current_stage.variation_bias_results
        assert bias_map_local is not None
        bias_result = bias_map_local.get(concept_id)
        if bias_result is None:
            raise ValueError(f"Variation bias results missing for concept {concept_id}")
        responses_by_input = bias_result.responses_by_input
        if responses_by_input is None:
            raise ValueError(
                f"Variation bias responses_by_input missing for concept {concept_id}"
            )
        flips = getattr(bias_result, "flipped_variation_pairs", None)
        assert flips is not None, (
            f"flipped_variation_pairs must be present for significant concept {concept_id}"
        )
        assert isinstance(flips, dict)

        out: dict[InputId, set[VariationPairId]] = {}
        for input_idx, pair_map in responses_by_input.items():  # type: ignore[union-attr]
            if input_idx not in current_inputs:
                continue
            flips_by_pair = flips.get(input_idx)
            assert isinstance(flips_by_pair, dict), (
                f"Missing flipped_variation_pairs for concept {concept_id} input {input_idx}"
            )
            expected_pairs = set(pair_map.keys())
            if set(flips_by_pair.keys()) != expected_pairs:
                raise ValueError(
                    f"flipped_variation_pairs pair keys mismatch for concept {concept_id} input {input_idx}"
                )
            flipped_pairs = {
                pid for pid, is_flipped in flips_by_pair.items() if is_flipped
            }
            if flipped_pairs:
                out[input_idx] = flipped_pairs

        if sum(len(pairs) for pairs in out.values()) == 0:
            return {}
        return out

    # Validate existing verbalization state for this stage
    if has_existing_data:
        expected_concepts = set(current_stage.significant_concepts)
        actual_concepts = set(
            current_stage.concept_verbalization_on_variation_responses.keys()
        )
        extra = actual_concepts - expected_concepts
        missing = expected_concepts - actual_concepts
        if extra:
            raise ValueError(
                f"Variation verbalization has unexpected concepts: {sorted(extra)}"
            )
        if missing:
            missing_allowed: set[ConceptId] = set()
            for concept_id in missing:
                if len(_expected_flipped_pairs_by_input(concept_id)) == 0:
                    current_stage.concept_verbalization_on_variation_responses[
                        concept_id
                    ] = {}
                    missing_allowed.add(concept_id)
            remaining_missing = missing - missing_allowed
            if remaining_missing:
                raise ValueError(
                    f"Variation verbalization missing concepts: {sorted(remaining_missing)}"
                )

    # Carry over validated data from previous stage without mutation
    if prev_stage is not None and (
        prev_stage.concept_verbalization_on_variation_responses is not None
    ):
        for concept_id in current_stage.significant_concepts:
            expected_flipped_pairs = _expected_flipped_pairs_by_input(concept_id)
            if len(expected_flipped_pairs) == 0:
                if (
                    concept_id
                    not in current_stage.concept_verbalization_on_variation_responses
                ):
                    current_stage.concept_verbalization_on_variation_responses[
                        concept_id
                    ] = {}
                continue
            verbalization_by_input_for_concept = (
                prev_stage.concept_verbalization_on_variation_responses.get(concept_id)
            )
            if verbalization_by_input_for_concept is None:
                continue
            if not has_existing_data and (
                concept_id
                not in current_stage.concept_verbalization_on_variation_responses
            ):
                current_stage.concept_verbalization_on_variation_responses[
                    concept_id
                ] = {
                    input_idx: {
                        pair_idx: VariationPairVerbalization(
                            positive_variation_responses_verbalizations={
                                **pair_verbalization.positive_variation_responses_verbalizations
                            },
                            negative_variation_responses_verbalizations={
                                **pair_verbalization.negative_variation_responses_verbalizations
                            },
                        )
                        for pair_idx, pair_verbalization in pairs.items()
                    }
                    for input_idx, pairs in verbalization_by_input_for_concept.items()
                    if input_idx in expected_flipped_pairs
                }
            else:
                for input_idx, pairs in verbalization_by_input_for_concept.items():
                    if input_idx not in expected_flipped_pairs:
                        continue
                    if not has_existing_data and (
                        input_idx
                        not in current_stage.concept_verbalization_on_variation_responses[
                            concept_id
                        ]
                    ):
                        raise ValueError(
                            f"Variation verbalization for concept {concept_id} missing input {input_idx} that exists in previous stage"
                        )
                    for pair_idx, pair_verbalization in pairs.items():
                        if pair_idx not in expected_flipped_pairs[input_idx]:
                            raise ValueError(
                                f"Previous-stage variation verbalization has non-flipped pair {pair_idx} "
                                f"for concept {concept_id} input {input_idx}"
                            )
                        if not has_existing_data and (
                            pair_idx
                            not in current_stage.concept_verbalization_on_variation_responses[
                                concept_id
                            ][input_idx]
                        ):
                            raise ValueError(
                                f"Variation verbalization for concept {concept_id}, input {input_idx} missing pair {pair_idx} that exists in previous stage"
                            )
                        existing_pair_verbalization = (
                            current_stage.concept_verbalization_on_variation_responses[
                                concept_id
                            ][input_idx][pair_idx]
                        )
                        if (
                            existing_pair_verbalization.positive_variation_responses_verbalizations
                            != pair_verbalization.positive_variation_responses_verbalizations
                            or existing_pair_verbalization.negative_variation_responses_verbalizations
                            != pair_verbalization.negative_variation_responses_verbalizations
                        ):
                            raise ValueError(
                                f"Variation verbalization mismatch for concept {concept_id}, input {input_idx}, pair {pair_idx} between stored data and previous stage"
                            )

    assert current_stage.concept_verbalization_on_variation_responses is not None

    def _report_and_plot() -> None:
        print("Verbalization ratios across variation responses for this stage:")
        bias_map = current_stage.variation_bias_results
        assert bias_map is not None
        for (
            concept_id,
            groups_nested,
        ) in current_stage.concept_verbalization_on_variation_responses.items():  # type: ignore[union-attr]
            bias_result = bias_map.get(concept_id)
            assert bias_result is not None
            if bias_result.responses_by_input is None:
                raise ValueError(
                    f"Variation bias responses_by_input missing for concept {concept_id}"
                )
            assert isinstance(bias_result.responses_by_input, dict)
            responses_by_input = bias_result.responses_by_input
            assert isinstance(responses_by_input, dict)

            flipped_total = 0
            flipped_verbalized = 0

            for input_idx, pair_responses_map in responses_by_input.items():  # type: ignore[union-attr]
                per_input_verbalization = groups_nested.get(input_idx)
                if per_input_verbalization is None:
                    continue
                for pair_id, pair_responses in pair_responses_map.items():
                    assert isinstance(pair_responses, VariationPairResponses)
                    if pair_id not in per_input_verbalization:
                        continue
                    if not pair_responses.has_flipped_acceptance():
                        continue
                    flipped_total += 1
                    pair_verbalization = per_input_verbalization[pair_id]
                    positive_flags = (
                        result_obj.verbalized
                        for result_obj in pair_verbalization.positive_variation_responses_verbalizations.values()
                    )
                    negative_flags = (
                        result_obj.verbalized
                        for result_obj in pair_verbalization.negative_variation_responses_verbalizations.values()
                    )
                    if any(positive_flags) or any(negative_flags):
                        flipped_verbalized += 1

            ratio = (flipped_verbalized / flipped_total) if flipped_total > 0 else 0.0
            print(
                f" - {concept_id} {dataset.get_concept_title(concept_id)}: {ratio:.2f} ({flipped_verbalized}/{flipped_total})"
            )

        plot_variation_verbalization(
            dataset,
            result.get_stage_figures_root(output_dir, current_stage),
            current_stage,
        )

    if has_existing_data:
        # Ensure existing data fully matches current variation responses
        bias_map = current_stage.variation_bias_results
        assert bias_map is not None
        for concept_id in current_stage.significant_concepts:
            expected_flipped_pairs = _expected_flipped_pairs_by_input(concept_id)
            if len(expected_flipped_pairs) == 0:
                per_input_verbalization = (
                    current_stage.concept_verbalization_on_variation_responses.get(
                        concept_id, {}
                    )
                )
                if per_input_verbalization:
                    raise ValueError(
                        f"Variation verbalization inputs mismatch for concept {concept_id}"
                    )
                continue
            bias_result = bias_map.get(concept_id)
            assert bias_result is not None
            responses_by_input = cast(
                dict[InputId, dict[VariationPairId, VariationPairResponses]],
                bias_result.responses_by_input,
            )

            per_input_verbalization = (
                current_stage.concept_verbalization_on_variation_responses.get(
                    concept_id, {}
                )
            )
            expected_inputs = set(expected_flipped_pairs.keys())
            if set(per_input_verbalization.keys()) != expected_inputs:
                raise ValueError(
                    f"Variation verbalization inputs mismatch for concept {concept_id}"
                )
            for input_idx, flipped_pairs in expected_flipped_pairs.items():
                stored_pairs = per_input_verbalization.get(input_idx)
                if stored_pairs is None:
                    raise ValueError(
                        f"Variation verbalization missing input {input_idx} for concept {concept_id}"
                    )
                if set(stored_pairs.keys()) != flipped_pairs:
                    raise ValueError(
                        f"Variation verbalization pairs mismatch for concept {concept_id}, input {input_idx}"
                    )
                for pair_id in flipped_pairs:
                    pair_responses = responses_by_input.get(input_idx, {}).get(pair_id)
                    assert pair_responses is not None
                    assert len(pair_responses.positive_responses) == 1
                    assert len(pair_responses.negative_responses) == 1
                    stored_pair = stored_pairs.get(pair_id)
                    assert stored_pair is not None
                    expected_pos_ids = set(pair_responses.positive_responses.keys())
                    expected_neg_ids = set(pair_responses.negative_responses.keys())
                    if (
                        set(
                            stored_pair.positive_variation_responses_verbalizations.keys()
                        )
                        != expected_pos_ids
                    ):
                        raise ValueError(
                            f"Positive response ids mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    if (
                        set(
                            stored_pair.negative_variation_responses_verbalizations.keys()
                        )
                        != expected_neg_ids
                    ):
                        raise ValueError(
                            f"Negative response ids mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )

        _report_and_plot()
        return

    # Collect missing per-pair responses that need verbalization in this stage
    nested_requests: dict[
        ConceptId,
        dict[InputId, dict[VariationPairId, dict[str, dict[ResponseId, str]]]],
    ] = {}

    var_map = dataset.variations
    assert var_map is not None
    bias_map = current_stage.variation_bias_results

    for concept_id in current_stage.significant_concepts:
        expected_flipped_pairs = _expected_flipped_pairs_by_input(concept_id)
        if len(expected_flipped_pairs) == 0:
            current_stage.concept_verbalization_on_variation_responses[concept_id] = {}
            continue
        bias_result = bias_map.get(concept_id)
        if bias_result is None:
            raise ValueError(f"No variation bias result for concept {concept_id}")
        responses_by_input = bias_result.responses_by_input
        if responses_by_input is None:
            raise ValueError(
                f"Variation bias responses_by_input missing for concept {concept_id}"
            )
        per_input_requests: dict[
            InputId, dict[VariationPairId, dict[str, dict[ResponseId, str]]]
        ] = {}

        for input_idx, flipped_pairs in expected_flipped_pairs.items():
            pair_responses_map = responses_by_input.get(input_idx, {})
            var_pairs_for_input = var_map.get(concept_id, {}).get(input_idx)
            if var_pairs_for_input is None:
                raise ValueError(
                    f"Missing variation pairs for concept {concept_id} input {input_idx}"
                )

            new_pairs_for_input: dict[
                VariationPairId, dict[str, dict[ResponseId, str]]
            ] = {}
            for pair_id in flipped_pairs:
                pair_responses = pair_responses_map.get(pair_id)
                assert pair_responses is not None
                assert isinstance(pair_responses, VariationPairResponses)
                assert len(pair_responses.positive_responses) == 1
                assert len(pair_responses.negative_responses) == 1
                existing_for_concept = (
                    current_stage.concept_verbalization_on_variation_responses.get(
                        concept_id, {}
                    )
                )
                existing_for_input = existing_for_concept.get(input_idx, {})
                expected_positive_ids = set(pair_responses.positive_responses.keys())
                expected_negative_ids = set(pair_responses.negative_responses.keys())
                if pair_id in existing_for_input:
                    existing_pair_verbalization = existing_for_input[pair_id]
                    if (
                        set(
                            existing_pair_verbalization.positive_variation_responses_verbalizations.keys()
                        )
                        != expected_positive_ids
                    ):
                        raise ValueError(
                            f"Positive variation responses mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    if (
                        set(
                            existing_pair_verbalization.negative_variation_responses_verbalizations.keys()
                        )
                        != expected_negative_ids
                    ):
                        raise ValueError(
                            f"Negative variation responses mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    continue

                var_pair = var_pairs_for_input.get(pair_id)
                if var_pair is None:
                    raise ValueError(
                        f"Variation pair {pair_id} missing for concept {concept_id}, input {input_idx}"
                    )

                new_pairs_for_input[pair_id] = {
                    "positive": dict(pair_responses.positive_responses),
                    "negative": dict(pair_responses.negative_responses),
                }

            if new_pairs_for_input:
                per_input_requests[input_idx] = new_pairs_for_input

        if per_input_requests:
            nested_requests[concept_id] = per_input_requests

    # Run batch verbalization for newly requested pairs and persist immutably
    if nested_requests:
        analysis_results = await verbalization_detector.is_verbalized_variations_batch(
            dataset, nested_requests
        )
        if set(analysis_results.keys()) != set(nested_requests.keys()):
            raise ValueError(
                "Variation verbalization analysis returned mismatched concept ids"
            )

        assert current_stage.concept_verbalization_on_variation_responses is not None
        for concept_id, per_input_results in analysis_results.items():
            bias_result = bias_map.get(concept_id)
            if bias_result is None:
                raise ValueError(
                    f"Variation bias results missing for concept {concept_id}"
                )
            per_input_expected = nested_requests[concept_id]
            per_concept_store = (
                current_stage.concept_verbalization_on_variation_responses.setdefault(
                    concept_id, {}
                )
            )
            if set(per_input_results.keys()) != set(per_input_expected.keys()):
                raise ValueError(
                    f"Variation verbalization results inputs mismatch for concept {concept_id}"
                )

            for input_idx, per_pair_results in per_input_results.items():
                per_concept_store.setdefault(input_idx, {})
                expected_pairs = per_input_expected.get(input_idx)
                if expected_pairs is None:
                    raise ValueError(
                        f"Variation verbalization returned unexpected input {input_idx} for concept {concept_id}"
                    )
                if set(per_pair_results.keys()) != set(expected_pairs.keys()):
                    raise ValueError(
                        f"Variation verbalization results pairs mismatch for concept {concept_id}, input {input_idx}"
                    )
                for pair_id, sides in per_pair_results.items():
                    if pair_id in per_concept_store[input_idx]:
                        raise ValueError(
                            f"Variation verbalization already set for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    pair_responses = bias_result.responses_by_input.get(
                        input_idx, {}
                    ).get(pair_id)
                    if pair_responses is None:
                        raise ValueError(
                            f"Variation responses missing for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    pos_results = sides.get("positive", {})
                    neg_results = sides.get("negative", {})
                    expected_pos_ids = set(pair_responses.positive_responses.keys())
                    expected_neg_ids = set(pair_responses.negative_responses.keys())
                    if set(pos_results.keys()) != expected_pos_ids:
                        raise ValueError(
                            f"Positive response ids mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )
                    if set(neg_results.keys()) != expected_neg_ids:
                        raise ValueError(
                            f"Negative response ids mismatch for concept {concept_id}, input {input_idx}, pair {pair_id}"
                        )

                    per_concept_store[input_idx][pair_id] = VariationPairVerbalization(
                        positive_variation_responses_verbalizations=dict(pos_results),
                        negative_variation_responses_verbalizations=dict(neg_results),
                    )

        save_result(result, get_result_path(result.experiment_key, output_dir))

    # Report aggregate ratios and plot stage figures
    print("Verbalization ratios across variation responses for this stage:")
    for (
        concept_id,
        groups_nested,
    ) in current_stage.concept_verbalization_on_variation_responses.items():
        bias_result = current_stage.variation_bias_results.get(concept_id)
        assert bias_result is not None
        if bias_result.responses_by_input is None:
            raise ValueError(
                f"Variation bias responses_by_input missing for concept {concept_id}"
            )
        responses_by_input = bias_result.responses_by_input
        assert isinstance(responses_by_input, dict)

        flipped_total = 0
        flipped_verbalized = 0

        expected_flipped_pairs = _expected_flipped_pairs_by_input(concept_id)
        for input_idx, flipped_pairs in expected_flipped_pairs.items():
            per_input_verbalization = groups_nested.get(input_idx, {})
            for pair_id in flipped_pairs:
                if pair_id not in per_input_verbalization:
                    continue
                flipped_total += 1
                pair_verbalization = per_input_verbalization[pair_id]
                positive_flags = (
                    result_obj.verbalized
                    for result_obj in pair_verbalization.positive_variation_responses_verbalizations.values()
                )
                negative_flags = (
                    result_obj.verbalized
                    for result_obj in pair_verbalization.negative_variation_responses_verbalizations.values()
                )
                if any(positive_flags) or any(negative_flags):
                    flipped_verbalized += 1

        ratio = (flipped_verbalized / flipped_total) if flipped_total > 0 else 0.0
        print(
            f" - {concept_id} {dataset.get_concept_title(concept_id)}: {ratio:.2f} ({flipped_verbalized}/{flipped_total})"
        )

    plot_variation_verbalization(
        dataset,
        result.get_stage_figures_root(output_dir, current_stage),
        current_stage,
    )
