from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)

from .concept_pipeline_result import ConceptPipelineResult, StageResults


def figure_width_from_num_xticks(
    num_xticks: int,
    base_num_xticks: int = 20,
    base_width: float = 12.0,
    slope_width_per_tick: float = 0.2,
    min_width: float = 8.0,
) -> float:
    """Compute figure width linearly from number of x-ticks.

    For `base_num_xticks`, the width equals `base_width`. For larger numbers, width
    grows linearly with slope `slope_width_per_tick`. For fewer ticks, the width
    does not shrink below `min_width`.

    This keeps prior behavior (12 inches around 20 concepts) while scaling to
    hundreds of concepts.
    """
    assert isinstance(num_xticks, int) and num_xticks >= 1
    assert base_num_xticks >= 1 and base_width > 0.0 and slope_width_per_tick >= 0.0
    assert min_width > 0.0
    extra_ticks = max(0, num_xticks - base_num_xticks)
    width = base_width + slope_width_per_tick * float(extra_ticks)
    return max(min_width, width)


def plot_bias_impact(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    figures_root: Path,
    stage: StageResults,
    model_name: str | None = None,
) -> Path:
    """Create and save the bias impact bar plot as bias_impact.png.

    Uses stage-based variation_bias_results (keyed by ConceptSide) to compute
    acceptance rates per concept side. Plots one bar per concept side with
    concept-side labels as x-ticks (no aggregation).
    """
    assert stage.variation_bias_results is not None and isinstance(
        stage.variation_bias_results, dict
    )

    os.makedirs(figures_root, exist_ok=True)

    labels: list[str] = []
    rates: list[float] = []
    valid_counts: list[int] = []
    total_counts: list[int] = []

    for concept_id, res in stage.variation_bias_results.items():
        acc = res.flatten_acceptances()
        valid = [v for v in acc if v is not None]
        rate = (
            float(sum(1 for v in valid if v == 1)) / float(len(valid))
            if len(valid) > 0
            else np.nan
        )
        labels.append(f"{dataset.get_concept_title(concept_id)}")
        rates.append(rate)
        valid_counts.append(len(valid))
        total_counts.append(len(acc))

    # If there are no records at all, still emit an empty-but-valid axis with title.
    if len(labels) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        title_suffix = (
            f"\n{model_name}"
            if isinstance(model_name, str) and len(model_name) > 0
            else ""
        )
        ax.set_axis_off()
        ax.set_title(
            f"Bias Impact of Generated Concepts{title_suffix}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        out_path = figures_root / "bias_impact.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Sort by acceptance rate descending (NaNs last)
    order = np.argsort(np.array([(-r if np.isfinite(r) else np.inf) for r in rates]))
    labels_sorted = [labels[i] for i in order]
    rates_sorted = [rates[i] for i in order]
    valid_sorted = [valid_counts[i] for i in order]
    total_sorted = [total_counts[i] for i in order]

    # Compute baseline first (used regardless of layout)
    baseline_acceptances = result.get_baseline_acceptances_list()
    baseline_valid = [a for a in baseline_acceptances if a is not None]
    assert len(baseline_valid) > 0
    baseline_rate = float(sum(1 for a in baseline_valid if a == 1)) / float(
        len(baseline_valid)
    )
    x = np.arange(len(labels_sorted))
    fig, ax = plt.subplots(
        figsize=(figure_width_from_num_xticks(len(labels_sorted)), 8)
    )
    bars = ax.bar(
        x,
        rates_sorted,
        0.8,
        label="Acceptance Rate",
        color="#2E86AB",
    )

    ax.set_ylabel("Acceptance Rate", fontsize=12)
    title_suffix = (
        f"\n{model_name}" if isinstance(model_name, str) and len(model_name) > 0 else ""
    )
    ax.set_title(
        f"Bias Impact of Generated Concepts{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")

    # Baseline average acceptance
    ax.axhline(
        baseline_rate,
        color="#6C6F7D",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline Avg ({baseline_rate:.2f})",
    )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels: pair bars with their (rate, valid_count, total_count)
    assert len(rates_sorted) == len(valid_sorted) == len(total_sorted) == len(bars)
    for i, bar in enumerate(bars):
        val = rates_sorted[i]
        vcnt = valid_sorted[i]
        tcnt = total_sorted[i]
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.2f}  ({vcnt}/{tcnt})",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=9,
            )

    plt.tight_layout()
    out_path = figures_root / "bias_impact.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stage_drop_reasons(
    result: ConceptPipelineResult,
    *,
    output_dir: Path,
) -> Path:
    """Plot stacked bars for per-stage drop reasons.

    Reasons:
    - Baseline verbalization check
    - Futility (conditional power)
    - Variation verbalization check (pair-level verbalization over flipped pairs)
    """
    assert isinstance(output_dir, Path)
    assert result.stages is not None and len(result.stages) > 0

    figures_root = result.get_figures_root(output_dir)
    os.makedirs(figures_root, exist_ok=True)

    stages = result.stages
    assert stages is not None

    stage_labels: list[str] = []
    dropped_baseline: list[int] = []
    dropped_futility: list[int] = []
    dropped_variation_verbalization: list[int] = []

    for stage in stages:
        stage_idx = stage.stage_idx
        stage_labels.append(f"Stage {stage_idx}")

        assert stage.concepts_at_stage_start is not None
        n_start = len(stage.concepts_at_stage_start)
        assert n_start >= 0

        assert stage.concept_ids_unverbalized_on_baseline is not None, (
            "Expected concept_ids_unverbalized_on_baseline to be present to compute baseline drop counts"
        )
        n_unverbalized = len(stage.concept_ids_unverbalized_on_baseline)
        assert 0 <= n_unverbalized <= n_start
        dropped_baseline.append(n_start - n_unverbalized)

        futility = stage.futility_stopped_concepts or []
        dropped_futility.append(len(futility))

        # Variation verbalization check drops among concepts that were evaluated post-futility.
        # In the pipeline, `significant_concepts` is computed after futility filtering.
        if stage.significant_concepts is None or stage.concepts_at_stage_end is None:
            dropped_variation_verbalization.append(0)
        else:
            n_sig = len(stage.significant_concepts)
            n_end = len(stage.concepts_at_stage_end)
            assert 0 <= n_end <= n_sig
            dropped_variation_verbalization.append(n_sig - n_end)

    x = np.arange(len(stage_labels))
    width = 0.75

    # Colors mimicking dashboard palette
    col_baseline = "#fff3e0"
    edge_baseline = "#ef6c00"
    col_futility = "#ffccbc"
    edge_futility = "#d84315"
    col_varverb = "#f3e5f5"
    edge_varverb = "#7b1fa2"

    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(stage_labels)), 5.0))

    b0 = np.array(dropped_baseline, dtype=np.int64)
    b1 = np.array(dropped_futility, dtype=np.int64)
    b2 = np.array(dropped_variation_verbalization, dtype=np.int64)
    assert b0.shape == b1.shape == b2.shape == (len(stage_labels),)

    ax.bar(
        x,
        b0,
        width,
        label="Dropped: baseline verbalization",
        color=col_baseline,
        edgecolor=edge_baseline,
        linewidth=1.5,
    )
    ax.bar(
        x,
        b1,
        width,
        bottom=b0,
        label="Dropped: futility",
        color=col_futility,
        edgecolor=edge_futility,
        linewidth=1.5,
    )
    ax.bar(
        x,
        b2,
        width,
        bottom=b0 + b1,
        label="Dropped: variation verbalization",
        color=col_varverb,
        edgecolor=edge_varverb,
        linewidth=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=0)
    ax.set_ylabel("Number of concepts dropped")
    ax.set_title("Concept drops by stage and reason", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    # Annotate totals on top of each bar
    totals = (b0 + b1 + b2).astype(np.int64)
    for xi, total in zip(x, totals, strict=False):
        ax.text(
            xi,
            float(total) + 0.2,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out_path = figures_root / "drop_reasons_by_stage.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_concept_baseline_verbalization(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    figures_root: Path,
    stage: StageResults,
) -> Path:
    """Plot average baseline verbalization rate per concept for a specific stage.

    Uses `stage.concept_verbalization_on_baseline_responses` strictly.
    """
    assert stage.concept_verbalization_on_baseline_responses is not None
    data = stage.concept_verbalization_on_baseline_responses
    assert isinstance(data, dict) and len(data) > 0

    os.makedirs(figures_root, exist_ok=True)

    concepts: list[str] = []
    averages: list[float] = []
    counts: list[tuple[int, int]] = []
    for concept_id, flags_by_input in data.items():
        assert isinstance(flags_by_input, dict) and len(flags_by_input) > 0
        all_flags: list[bool] = []
        for flags_map in flags_by_input.values():
            assert isinstance(flags_map, dict) and len(flags_map) > 0
            for result_obj in flags_map.values():
                all_flags.append(result_obj.verbalized)
        total = len(all_flags)
        assert total > 0
        positives = sum(1 for v in all_flags if v)
        avg = positives / total
        concepts.append(dataset.get_concept_title(concept_id))
        averages.append(avg)
        counts.append((positives, total))

    # Sort by average verbalization descending
    order = np.argsort(np.array(averages))[::-1]
    concepts_sorted = [concepts[i] for i in order]
    averages_sorted = [averages[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    x = np.arange(len(concepts_sorted))
    fig, ax = plt.subplots(
        figsize=(figure_width_from_num_xticks(len(concepts_sorted)), 8)
    )
    bars = ax.bar(x, averages_sorted, color="#3B8EA5")
    ax.set_ylabel("Avg. Verbalization Rate", fontsize=12)
    ax.set_title("Baseline Concept Verbalization", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts_sorted, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Annotate with counts like 5/10
    for bar, (pos, tot) in zip(bars, counts_sorted, strict=False):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{pos}/{tot}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=9,
            )

    plt.tight_layout()
    out_path = figures_root / "concept_baseline_verbalization.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pvalues(
    dataset: ConceptPipelineDataset,
    figures_root: Path,
    stage: StageResults,
    significance_test: str,
) -> Path:
    """Plot p-values per concept for positive vs negative comparison.

    Args:
        result: The pipeline result containing concept information
        figures_root: Directory to save the plot
        stage: The stage results containing variation bias results
        significance_test: Either "fisher" or "mcnemar" to determine which test's p-values to plot

    Returns:
        Path to the saved plot
    """
    assert stage.variation_bias_results is not None
    thr = stage.early_stop_alpha
    assert isinstance(thr, float) and 0.0 <= thr <= 1.0
    assert significance_test in ("fisher", "mcnemar"), (
        f"Invalid significance test: {significance_test}"
    )

    results = stage.variation_bias_results
    os.makedirs(figures_root, exist_ok=True)

    # Determine the p-value key and test name based on significance_test parameter
    pval_key = "mcnemar_p_value" if significance_test == "mcnemar" else "fisher_p_value"
    test_name = "McNemar" if significance_test == "mcnemar" else "Fisher Exact"
    filename = f"{significance_test}_pvalues_positive_vs_negative.png"

    labels: list[str] = []
    pvalues: list[float] = []

    # Gather concepts (not concept sides) and their positive vs negative p-values
    for concept_id, res in results.items():
        stats = res.statistics_positive_vs_negative
        if not (isinstance(stats, dict) and pval_key in stats):
            continue
        p_value = stats[pval_key]
        if not (isinstance(p_value, float) and 0.0 <= p_value <= 1.0):
            continue
        labels.append(dataset.get_concept_title(concept_id))
        pvalues.append(p_value)

    if len(labels) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        title = f"{test_name} Test p-values: Positive vs Negative"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        out_path = figures_root / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    order = np.argsort(np.array(pvalues))
    labels_sorted = [labels[i] for i in order]
    pvalues_sorted = [pvalues[i] for i in order]

    x = np.arange(len(labels_sorted))
    fig, ax = plt.subplots(
        figsize=(figure_width_from_num_xticks(len(labels_sorted)), 8)
    )
    bars = ax.bar(x, pvalues_sorted, color="#7D823A")
    ax.set_ylabel(f"{test_name} p-value", fontsize=12)
    ax.set_title(
        f"{test_name} Test p-values: Positive vs Negative",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    ax.axhline(
        thr,
        color="#D35400",
        linestyle="--",
        linewidth=1.5,
        label=f"O'Brien-Fleming α ({thr:.3g})",
    )
    ax.legend()

    for bar, p in zip(bars, pvalues_sorted, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{p:.3g}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9,
        )

    plt.tight_layout()
    out_path = figures_root / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_histogram(
    *,
    values: list[float],
    threshold: float,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
) -> Path:
    assert isinstance(values, list)
    assert len(values) > 0
    assert isinstance(threshold, float) and 0.0 <= threshold <= 1.0
    assert isinstance(title, str) and len(title) > 0
    assert isinstance(xlabel, str) and len(xlabel) > 0
    assert isinstance(out_path, Path)
    assert isinstance(bins, int) and bins >= 5

    arr = np.array(values, dtype=np.float64)
    assert arr.ndim == 1 and arr.shape[0] == len(values)
    assert np.all(np.isfinite(arr)), "Histogram values must be finite"
    assert np.all((0.0 <= arr) & (arr <= 1.0)), "Histogram values must be in [0,1]"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(arr, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
    ax.axvline(
        threshold,
        color="#D35400",
        linestyle="--",
        linewidth=2.0,
        label=f"threshold={threshold:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_histogram_no_threshold(
    *,
    values: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
) -> Path:
    assert isinstance(values, list)
    assert len(values) > 0
    assert isinstance(title, str) and len(title) > 0
    assert isinstance(xlabel, str) and len(xlabel) > 0
    assert isinstance(out_path, Path)
    assert isinstance(bins, int) and bins >= 5

    arr = np.array(values, dtype=np.float64)
    assert arr.ndim == 1 and arr.shape[0] == len(values)
    assert np.all(np.isfinite(arr)), "Histogram values must be finite"
    assert np.all((0.0 <= arr) & (arr <= 1.0)), "Histogram values must be in [0,1]"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(arr, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stage_histograms(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    stage: StageResults,
    *,
    output_dir: Path,
) -> None:
    """Write per-stage histograms for pipeline diagnostics.

    Always writes the baseline verbalization histogram.
    Conditionally writes conditional power and pair-variation verbalization histograms
    if the required per-concept data is present on the stage.
    """
    assert isinstance(output_dir, Path)
    figures_root = result.get_stage_figures_root(output_dir, stage)
    stage_idx = stage.stage_idx

    # 1) Baseline verbalization per concept (stage 0 only, if present)
    if (
        result.baseline_verbalization_threshold is not None
        and stage.concept_verbalization_on_baseline_responses is not None
    ):
        base_thr = float(result.baseline_verbalization_threshold)
        assert (
            stage.concepts_at_stage_start is not None
            and len(stage.concepts_at_stage_start) > 0
        )
        base_flags = stage.concept_verbalization_on_baseline_responses
        assert isinstance(base_flags, dict) and len(base_flags) > 0

        baseline_scores: list[float] = []
        for concept_id in stage.concepts_at_stage_start:
            by_input = base_flags.get(concept_id)
            assert by_input is not None, (
                f"Missing baseline verbalization data for concept {concept_id} at stage {stage_idx}"
            )
            flags: list[bool] = []
            for by_resp in by_input.values():
                for res in by_resp.values():
                    flags.append(bool(res.verbalized))
            assert len(flags) > 0
            baseline_scores.append(float(sum(1 for f in flags if f) / len(flags)))

        _plot_histogram(
            values=baseline_scores,
            threshold=base_thr,
            title=f"Stage {stage_idx} — baseline verbalization (per concept)",
            xlabel="Baseline verbalization score",
            out_path=figures_root / "hist_baseline_verbalization.png",
        )

    # 1b) Flipped pair rate per concept (fraction of variation pairs that flip acceptances)
    if (
        stage.variation_bias_results is not None
        and len(stage.variation_bias_results) > 0
    ):
        flipped_rates: list[float] = []
        for _concept_id, bias_res in stage.variation_bias_results.items():
            total_pairs = sum(
                len(by_pair) for by_pair in bias_res.responses_by_input.values()
            )
            assert total_pairs > 0
            flips = getattr(bias_res, "flipped_variation_pairs", None)
            if flips is None:
                # Compute deterministically from acceptances (same semantics as VariationPairResponses.has_flipped_acceptance).
                flipped_pairs = 0
                for by_pair in bias_res.responses_by_input.values():
                    for pair_responses in by_pair.values():
                        if pair_responses.has_flipped_acceptance():
                            flipped_pairs += 1
            else:
                flipped_pairs = sum(
                    1
                    for per_input in flips.values()
                    for is_flipped in per_input.values()
                    if is_flipped
                )
            assert 0 <= flipped_pairs <= total_pairs
            flipped_rates.append(float(flipped_pairs / total_pairs))
        if len(flipped_rates) > 0:
            _plot_histogram_no_threshold(
                values=flipped_rates,
                title=f"Stage {stage_idx} — flipped pair rate (per concept)",
                xlabel="Fraction of flipped variation pairs",
                out_path=figures_root / "hist_flipped_pairs_rate.png",
            )

    # 2) Conditional power per concept (only if present)
    if (
        result.futility_stop_power_threshold is not None
        and stage.conditional_power_by_concept is not None
        and len(stage.conditional_power_by_concept) > 0
    ):
        fut_thr = float(result.futility_stop_power_threshold)
        cp_vals = list(stage.conditional_power_by_concept.values())
        _plot_histogram(
            values=[float(v) for v in cp_vals],
            threshold=fut_thr,
            title=f"Stage {stage_idx} — conditional power (per concept)",
            xlabel="Conditional power",
            out_path=figures_root / "hist_conditional_power.png",
        )

    # 3) Pair variation verbalization per concept (only if present)
    if (
        result.variations_verbalization_threshold is not None
        and stage.concept_verbalization_on_variation_responses is not None
        and stage.variation_bias_results is not None
        and len(stage.concept_verbalization_on_variation_responses) > 0
    ):
        var_thr = float(result.variations_verbalization_threshold)
        pair_scores: list[float] = []
        for (
            concept_id,
            by_input,
        ) in stage.concept_verbalization_on_variation_responses.items():
            bias_res = stage.variation_bias_results.get(concept_id)
            assert bias_res is not None, (
                f"Missing variation_bias_results for concept {concept_id} at stage {stage_idx}"
            )
            responses_by_input = bias_res.responses_by_input
            flipped_total = 0
            flipped_verbalized = 0

            for input_id, by_pair in responses_by_input.items():
                per_input_verb = by_input.get(input_id)
                if per_input_verb is None:
                    continue
                for pair_id, pair_responses in by_pair.items():
                    if pair_id not in per_input_verb:
                        continue
                    if not pair_responses.has_flipped_acceptance():
                        continue
                    flipped_total += 1
                    pair_verb = per_input_verb[pair_id]
                    pos_flags = (
                        r.verbalized
                        for r in pair_verb.positive_variation_responses_verbalizations.values()
                    )
                    neg_flags = (
                        r.verbalized
                        for r in pair_verb.negative_variation_responses_verbalizations.values()
                    )
                    if any(pos_flags) or any(neg_flags):
                        flipped_verbalized += 1

            if flipped_total == 0:
                continue
            pair_scores.append(float(flipped_verbalized / flipped_total))

        if len(pair_scores) > 0:
            _plot_histogram(
                values=pair_scores,
                threshold=var_thr,
                title=f"Stage {stage_idx} — pair variation verbalization (per concept)",
                xlabel="Pair variation verbalization",
                out_path=figures_root / "hist_pair_variation_verbalization.png",
            )


def plot_variation_verbalization(
    dataset: ConceptPipelineDataset,
    figures_root: Path,
    stage: StageResults,
) -> Path:
    """Plot verbalization rates on variation responses for each concept side."""
    assert stage.concept_verbalization_on_variation_responses is not None
    data = stage.concept_verbalization_on_variation_responses
    assert isinstance(data, dict) and len(data) > 0
    baseline_detail = stage.concept_verbalization_on_baseline_responses

    os.makedirs(figures_root, exist_ok=True)

    # Data is now keyed by ConceptId, not ConceptSide
    # If baseline verbalization is not present for this stage, plot variation-only.
    if baseline_detail is None:
        concept_ids = sorted(data.keys())
    else:
        assert isinstance(baseline_detail, dict)
        # Only plot concepts that have both variation and baseline data
        concept_ids = sorted(cid for cid in data.keys() if cid in baseline_detail)

    if not concept_ids:
        print(
            "Warning: No concepts have both baseline and variation verbalization data. Skipping plot."
        )
        # Return a dummy path
        return figures_root / "variation_verbalization.png"

    concepts: list[str] = []
    pos_rates: list[float] = []
    neg_rates: list[float] = []
    pos_counts: list[tuple[int, int]] = []
    neg_counts: list[tuple[int, int]] = []
    baseline_rates: list[float] = []
    baseline_counts: list[tuple[int, int]] = []

    for concept_id in concept_ids:
        # Get verbalization data for this concept
        concept_data = data[concept_id]

        pos_total = 0
        pos_pos = 0
        neg_total = 0
        neg_pos = 0

        # Iterate through inputs and pairs for this concept
        for per_input in concept_data.values():
            for per_pair in per_input.values():
                # per_pair is a VariationPairVerbalization object
                # with positive_variation_responses_verbalizations and negative_variation_responses_verbalizations
                assert hasattr(per_pair, "positive_variation_responses_verbalizations")
                assert hasattr(per_pair, "negative_variation_responses_verbalizations")

                pos_flags = list(
                    per_pair.positive_variation_responses_verbalizations.values()
                )
                neg_flags = list(
                    per_pair.negative_variation_responses_verbalizations.values()
                )

                pos_total += len(pos_flags)
                pos_pos += sum(1 for v in pos_flags if v.verbalized)
                neg_total += len(neg_flags)
                neg_pos += sum(1 for v in neg_flags if v.verbalized)

        pos_rate = (pos_pos / pos_total) if pos_total > 0 else 0.0
        neg_rate = (neg_pos / neg_total) if neg_total > 0 else 0.0

        concepts.append(dataset.get_concept_title(concept_id))
        pos_rates.append(pos_rate)
        neg_rates.append(neg_rate)
        pos_counts.append((pos_pos, pos_total))
        neg_counts.append((neg_pos, neg_total))

        if baseline_detail is not None:
            # We already filtered to only concepts in baseline_detail
            base_flags_by_input = baseline_detail[concept_id]
            assert (
                isinstance(base_flags_by_input, dict) and len(base_flags_by_input) > 0
            )
            base_flags_all: list[bool] = []
            for flags_map in base_flags_by_input.values():
                assert isinstance(flags_map, dict) and len(flags_map) > 0
                for result_obj in flags_map.values():
                    base_flags_all.append(result_obj.verbalized)
            base_total = len(base_flags_all)
            base_pos = sum(1 for v in base_flags_all if v)
            base_rate = base_pos / base_total
            baseline_rates.append(base_rate)
            baseline_counts.append((base_pos, base_total))

    # Sort concepts by positive rate descending, but plot per concept side as x-ticks
    order = np.argsort(np.array(pos_rates))[::-1]
    concepts_sorted = [concepts[i] for i in order]
    pos_sorted = [pos_rates[i] for i in order]
    neg_sorted = [neg_rates[i] for i in order]
    pos_counts_sorted = [pos_counts[i] for i in order]
    neg_counts_sorted = [neg_counts[i] for i in order]
    if baseline_detail is not None:
        assert len(baseline_rates) == len(concepts) == len(baseline_counts)
        base_sorted = [baseline_rates[i] for i in order]
        base_counts_sorted = [baseline_counts[i] for i in order]
    else:
        base_sorted = []
        base_counts_sorted = []

    # Build per-side arrays
    labels: list[str] = []
    side_vals: list[float] = []
    side_colors: list[str] = []
    side_counts: list[tuple[int, int]] = []
    base_vals: list[float] = []
    base_counts_vals: list[tuple[int, int]] = []
    for i, concept in enumerate(concepts_sorted):
        labels.append(f"{concept} (positive)")
        labels.append(f"{concept} (negative)")
        side_vals.extend([pos_sorted[i], neg_sorted[i]])
        side_colors.extend(["#2E86AB", "#A23B72"])  # positive, negative
        side_counts.extend([pos_counts_sorted[i], neg_counts_sorted[i]])
        if baseline_detail is not None:
            base_vals.extend([base_sorted[i], base_sorted[i]])
            base_counts_vals.extend([base_counts_sorted[i], base_counts_sorted[i]])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(figure_width_from_num_xticks(len(labels)), 8))
    bars_side = ax.bar(
        x - width / 2,
        side_vals,
        width,
        color=side_colors,
        label="Side",
    )
    bars_base = None
    if baseline_detail is not None:
        bars_base = ax.bar(
            x + width / 2,
            base_vals,
            width,
            color="#6C6F7D",
            label="Baseline",
        )

    ax.set_ylabel("Verbalization Rate", fontsize=12)
    ax.set_title("Verbalization on Variation Responses", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Legend with explicit side colors
    legend_handles = [
        Patch(color="#2E86AB", label="Positive"),
        Patch(color="#A23B72", label="Negative"),
    ]
    if baseline_detail is not None:
        legend_handles.append(Patch(color="#6C6F7D", label="Baseline"))
    ax.legend(handles=legend_handles)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Annotate with counts, e.g., 3/5 atop each bar
    for bar, (pos, tot) in zip(bars_side, side_counts, strict=False):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{pos}/{tot}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=9,
            )
    if bars_base is not None:
        for bar, (pos, tot) in zip(bars_base, base_counts_vals, strict=False):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{pos}/{tot}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                )

    plt.tight_layout()
    out_path = figures_root / "variation_verbalization.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
