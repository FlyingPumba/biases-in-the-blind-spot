from collections.abc import Callable
from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)


def start_new_stage(
    result: ConceptPipelineResult,
    stage_index: int,
    *,
    dataset: ConceptPipelineDataset,
    representative_inputs_k_per_stage_index_fn: Callable[[int], int],
    output_dir: Path,
) -> StageResults:
    assert dataset.input_clusters is not None
    assert result.filtered_varying_inputs is not None
    assert result.stages is not None

    concepts_in_last_stage = compute_concepts_in_last_stage(
        result, stage_index, dataset=dataset
    )

    k_inputs_per_representative_cluster = representative_inputs_k_per_stage_index_fn(
        stage_index
    )
    input_indices_by_representative_cluster = (
        dataset.select_inputs_from_each_representative_cluster(
            k_inputs_per_representative_cluster,
            output_dir,
            whitelisted_input_ids=result.filtered_varying_inputs,
        )
    )

    if len(result.stages) <= stage_index:
        current_stage = StageResults(
            stage_idx=stage_index,
            k_inputs_per_representative_cluster=k_inputs_per_representative_cluster,
            seed=42,
            input_indices_by_representative_cluster=input_indices_by_representative_cluster,
            concepts_at_stage_start=concepts_in_last_stage,
        )
        result.stages.append(current_stage)
    else:
        current_stage = result.stages[stage_index]
        assert current_stage is not None
        prev_stage = result.stages[stage_index - 1]
        assert prev_stage is not None
        if (
            current_stage.k_inputs_per_representative_cluster
            != k_inputs_per_representative_cluster
        ):
            raise ValueError(
                "k_inputs_per_representative_cluster mismatch for existing stage "
                f"{stage_index}: stored={current_stage.k_inputs_per_representative_cluster}, "
                f"requested={k_inputs_per_representative_cluster}"
            )
        if (
            current_stage.input_indices_by_representative_cluster
            != input_indices_by_representative_cluster
        ):
            raise ValueError(
                f"input_indices_by_representative_cluster mismatch for stage {stage_index}"
            )

    return current_stage


def compute_concepts_in_last_stage(
    result: ConceptPipelineResult,
    stage_index: int,
    *,
    dataset: ConceptPipelineDataset,
) -> list[ConceptId]:
    assert result.stages is not None
    concepts_in_last_stage: list[ConceptId]
    if stage_index == 0:
        pipeline_concepts = dataset.get_pipeline_concepts()
        concepts_in_last_stage = [c.id for c in pipeline_concepts]
    else:
        assert len(result.stages) > 0
        last_stage = result.stages[stage_index - 1]
        assert last_stage is not None, f"Stage {stage_index - 1} is not found"
        assert last_stage.concepts_at_stage_end is not None, (
            f"Concepts at stage end for stage {stage_index - 1} are not found"
        )
        concepts_in_last_stage = list(last_stage.concepts_at_stage_end)

        excluded: set[ConceptId] = set()
        if last_stage.early_stopped_concepts:
            excluded.update(last_stage.early_stopped_concepts)
        if last_stage.futility_stopped_concepts:
            excluded.update(last_stage.futility_stopped_concepts)

        if excluded:
            concepts_in_last_stage = [
                c for c in concepts_in_last_stage if c not in excluded
            ]
            print(
                f"Excluding {len(excluded)} stopped concepts from stage {stage_index} "
                f"(early={len(last_stage.early_stopped_concepts or [])}, "
                f"futility={len(last_stage.futility_stopped_concepts or [])})"
            )
    return concepts_in_last_stage


def compute_final_unfaithful_concepts(
    result: ConceptPipelineResult,
    last_stage: StageResults,
) -> None:
    assert result.stages is not None and len(result.stages) > 0

    all_early_stopped: list[ConceptId] = []
    for stage in result.stages:
        if stage.early_stopped_concepts:
            all_early_stopped.extend(stage.early_stopped_concepts)

    assert last_stage.early_stop_alpha is not None
    p_thr = float(last_stage.early_stop_alpha)
    use_mcnemar = result.significance_test == "mcnemar"
    significant_unfaithful: list[ConceptId] = []

    concept_ids_at_end = last_stage.concepts_at_stage_end or []
    for concept_id in concept_ids_at_end:
        assert last_stage.variation_bias_results is not None
        res = last_stage.variation_bias_results.get(concept_id)
        if res is None:
            continue
        stats = res.statistics_positive_vs_negative
        if isinstance(stats, dict):
            pval_key = "mcnemar_p_value" if use_mcnemar else "fisher_p_value"
            pval = stats.get(pval_key)
            if isinstance(pval, float) and pval < p_thr:
                significant_unfaithful.append(concept_id)

    significant_unfaithful.extend(all_early_stopped)

    seen = set()
    expected_significant_unfaithful = [
        c for c in significant_unfaithful if not (c in seen or seen.add(c))
    ]

    if result.significant_unfaithful_concepts is None:
        result.significant_unfaithful_concepts = expected_significant_unfaithful
    else:
        if result.significant_unfaithful_concepts != expected_significant_unfaithful:
            raise ValueError(
                "Existing significant_unfaithful_concepts does not match computed values"
            )

    print(
        f"\nFinal significant unfaithful concepts: {len(expected_significant_unfaithful)}"
    )
    print(
        f"  - From last stage: {len(significant_unfaithful) - len(all_early_stopped)}"
    )
    print(f"  - Early stopped (efficacy): {len(all_early_stopped)}")


def report_final_unfaithful_concepts_details(
    result: ConceptPipelineResult,
    dataset: ConceptPipelineDataset,
) -> None:
    """Print per-concept details for final significant unfaithful concepts.

    This uses the *latest stage where each concept was processed* (which may be
    earlier than the last stage if the concept was early-stopped for efficacy).
    """
    assert result.stages is not None and len(result.stages) > 0
    assert result.significant_unfaithful_concepts is not None
    assert result.significance_test in ("mcnemar", "fisher")

    def _find_latest_stage_with_bias_result(concept_id: ConceptId) -> StageResults:
        for stage in reversed(result.stages or []):
            if stage.variation_bias_results is not None and (
                concept_id in stage.variation_bias_results
            ):
                return stage
        raise ValueError(
            f"Could not find a stage with variation_bias_results for concept {concept_id}"
        )

    def _baseline_verbalization_ratio(
        stage: StageResults, concept_id: ConceptId
    ) -> tuple[float, int, int]:
        by_concept = stage.concept_verbalization_on_baseline_responses
        assert by_concept is not None
        by_input = by_concept.get(concept_id)
        if by_input is None:
            raise ValueError(
                f"Missing baseline verbalization data for concept {concept_id} in stage {stage.stage_idx}"
            )
        flags: list[bool] = [
            res.verbalized
            for per_input in by_input.values()
            for res in per_input.values()
        ]
        if len(flags) == 0:
            raise ValueError(
                f"Empty baseline verbalization flags for concept {concept_id} in stage {stage.stage_idx}"
            )
        positives = sum(1 for v in flags if v)
        return positives / len(flags), positives, len(flags)

    def _variation_verbalization_ratio(
        stage: StageResults, concept_id: ConceptId
    ) -> tuple[float, int, int]:
        by_concept = stage.concept_verbalization_on_variation_responses
        assert by_concept is not None
        groups_nested = by_concept.get(concept_id)
        if groups_nested is None:
            raise ValueError(
                f"Missing variation verbalization data for concept {concept_id} in stage {stage.stage_idx}"
            )

        assert stage.variation_bias_results is not None
        bias_result = stage.variation_bias_results.get(concept_id)
        assert bias_result is not None
        responses_by_input = bias_result.responses_by_input
        assert isinstance(responses_by_input, dict)

        flipped_total = 0
        flipped_verbalized = 0
        for input_idx, pair_responses_map in responses_by_input.items():
            per_input_verbalization = groups_nested.get(input_idx)
            if per_input_verbalization is None:
                continue
            for pair_id, pair_responses in pair_responses_map.items():
                if pair_id not in per_input_verbalization:
                    continue
                if not pair_responses.has_flipped_acceptance():
                    continue
                flipped_total += 1
                pair_verbalization = per_input_verbalization[pair_id]
                pos_any = any(
                    r.verbalized
                    for r in pair_verbalization.positive_variation_responses_verbalizations.values()
                )
                neg_any = any(
                    r.verbalized
                    for r in pair_verbalization.negative_variation_responses_verbalizations.values()
                )
                if pos_any or neg_any:
                    flipped_verbalized += 1

        ratio = (flipped_verbalized / flipped_total) if flipped_total > 0 else 0.0
        return ratio, flipped_verbalized, flipped_total

    concept_ids = list(result.significant_unfaithful_concepts)

    # Sort by strongest absolute bias strength first (descending).
    # Bias strength is defined as the per-stage proportion_difference.
    diffs_by_concept: dict[ConceptId, float] = {}
    for concept_id in concept_ids:
        stage = _find_latest_stage_with_bias_result(concept_id)
        assert stage.variation_bias_results is not None
        bias_res = stage.variation_bias_results[concept_id]
        stats = bias_res.statistics_positive_vs_negative
        if not isinstance(stats, dict):
            raise ValueError(
                f"Missing statistics_positive_vs_negative for concept {concept_id} in stage {stage.stage_idx}"
            )
        diff = stats.get("proportion_difference")
        assert isinstance(diff, float)
        diffs_by_concept[concept_id] = diff

    concept_ids.sort(key=lambda cid: abs(diffs_by_concept[cid]), reverse=True)
    print("\nFinal unfaithful concept details (latest processed stage per concept):")
    for concept_id in concept_ids:
        stage = _find_latest_stage_with_bias_result(concept_id)
        assert stage.variation_bias_results is not None
        bias_res = stage.variation_bias_results[concept_id]
        stats = bias_res.statistics_positive_vs_negative
        if not isinstance(stats, dict):
            raise ValueError(
                f"Missing statistics_positive_vs_negative for concept {concept_id} in stage {stage.stage_idx}"
            )

        diff = stats.get("proportion_difference")
        pos_rate = stats.get("positive_proportion")
        neg_rate = stats.get("negative_proportion")
        pval_key = (
            "mcnemar_p_value"
            if result.significance_test == "mcnemar"
            else "fisher_p_value"
        )
        pval = stats.get(pval_key)
        assert isinstance(diff, float)
        assert isinstance(pos_rate, float)
        assert isinstance(neg_rate, float)
        assert isinstance(pval, float)

        if stage.concept_verbalization_on_baseline_responses is None:
            base_ratio = None
            base_yes = None
            base_total = None
        else:
            base_ratio, base_yes, base_total = _baseline_verbalization_ratio(
                stage, concept_id
            )
        var_ratio, var_yes, var_total = _variation_verbalization_ratio(
            stage, concept_id
        )

        title = dataset.get_concept_title(concept_id)
        base_str = (
            "baseline_verbalization=N/A"
            if base_ratio is None
            else f"baseline_verbalization={base_ratio:.2f} ({base_yes}/{base_total})"
        )
        print(
            f" - {concept_id} - {title} "
            f"(stage={stage.stage_idx}): "
            f"bias_strength(diff)={diff:.3f} [pos={pos_rate:.3f}, neg={neg_rate:.3f}, p={pval:.3g}], "
            f"{base_str}, "
            f"variation_verbalization={var_ratio:.2f} ({var_yes}/{var_total})"
        )
