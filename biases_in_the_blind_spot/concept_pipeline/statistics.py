"""Early stopping analysis for sequential testing in concept pipeline.

Implements:
- O'Brien-Fleming alpha spending for efficacy stopping
- Conditional power for futility stopping
- McNemar's test matching the pipeline's implementation
"""

import math
import sys

import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar

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


def compute_mcnemar(b: int, c: int) -> tuple[float, float]:
    """Compute McNemar test statistic and p-value.

    Matches pipeline implementation: exact test for n<25, chi-squared for n≥25.

    Args:
        b: Count of discordant pairs (positive accept, negative reject)
        c: Count of discordant pairs (positive reject, negative accept)

    Returns:
        (statistic, p_value)
    """
    discordant = b + c
    if discordant == 0:
        # models never disagreed → no evidence of difference
        return 0.0, 1.0

    table = np.array([[0, b], [c, 0]])
    result = mcnemar(table, exact=(discordant < 25))
    return float(result.statistic), float(result.pvalue)  # type: ignore[attr-defined]


def compute_mcnemar_detailed(
    positive_acceptances: list[int | None],
    negative_acceptances: list[int | None],
) -> dict[str, float | None]:
    assert len(positive_acceptances) == len(negative_acceptances)

    paired: list[tuple[int, int]] = []
    for pos, neg in zip(positive_acceptances, negative_acceptances, strict=True):
        if pos is None or neg is None:
            continue
        paired.append((pos, neg))

    total_pairs = len(paired)
    if total_pairs == 0:
        return {
            "mcnemar_total_pairs": 0.0,
            "mcnemar_concordant_accept": 0.0,
            "mcnemar_concordant_reject": 0.0,
            "mcnemar_pos_accept_neg_reject": 0.0,
            "mcnemar_pos_reject_neg_accept": 0.0,
            "mcnemar_statistic": None,
            "mcnemar_p_value": None,
        }

    concordant_accept = sum(1 for pos, neg in paired if pos == 1 and neg == 1)
    concordant_reject = sum(1 for pos, neg in paired if pos == 0 and neg == 0)
    pos_accept_neg_reject = sum(1 for pos, neg in paired if pos == 1 and neg == 0)
    pos_reject_neg_accept = sum(1 for pos, neg in paired if pos == 0 and neg == 1)

    statistic, p_value = compute_mcnemar(pos_accept_neg_reject, pos_reject_neg_accept)

    return {
        "mcnemar_total_pairs": float(total_pairs),
        "mcnemar_concordant_accept": float(concordant_accept),
        "mcnemar_concordant_reject": float(concordant_reject),
        "mcnemar_pos_accept_neg_reject": float(pos_accept_neg_reject),
        "mcnemar_pos_reject_neg_accept": float(pos_reject_neg_accept),
        "mcnemar_statistic": statistic,
        "mcnemar_p_value": p_value,
    }


def compute_fisher_detailed(
    pos_accepted: int,
    pos_rejected: int,
    neg_accepted: int,
    neg_rejected: int,
) -> dict[str, float | None]:
    table = np.array([[pos_accepted, pos_rejected], [neg_accepted, neg_rejected]])
    assert table.shape == (2, 2)

    # For a 2x2 table, the null hypothesis is that the true odds ratio of the populations underlying the observations is one, and the observations were sampled from these populations under a condition: the marginals of the resulting table must equal those of the observed table. The statistic is the unconditional maximum likelihood estimate of the odds ratio, and the p-value is the probability under the null hypothesis of obtaining a table at least as extreme as the one that was actually observed.
    significance_result = fisher_exact(table)
    # res: SignificanceResult
    # An object containing attributes:
    #  - statistic: float
    #   For a 2x2 table with default method, this is the odds ratio - the prior odds ratio not a posterior estimate. In all other cases, this is the probability density of obtaining the observed table under the null hypothesis of independence with marginals fixed.
    #  - pvalue: float
    #   The probability under the null hypothesis of obtaining a table at least as extreme as the one that was actually observed.
    fisher_p_value: float = significance_result.pvalue  # type: ignore
    fisher_odds_ratio: float = significance_result.statistic  # type: ignore

    # Replace infinity with sys.maxsize for JSON compatibility
    if np.isinf(fisher_odds_ratio):
        fisher_odds_ratio = float(sys.maxsize)

    # 95% CI for Odds Ratio via log(OR) normal approximation; requires all cells > 0
    if pos_accepted > 0 and pos_rejected > 0 and neg_accepted > 0 and neg_rejected > 0:
        a = pos_accepted
        b = pos_rejected
        c = neg_accepted
        d = neg_rejected
        log_or = float(np.log(fisher_odds_ratio))
        se_log_or = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
        z_95 = 1.959963984540054
        or_ci_low = float(np.exp(log_or - z_95 * se_log_or))
        or_ci_high = float(np.exp(log_or + z_95 * se_log_or))
    else:
        or_ci_low = None
        or_ci_high = None

    return {
        "fisher_p_value": fisher_p_value,
        "fisher_odds_ratio": fisher_odds_ratio,
        "fisher_odds_ratio_ci_low": or_ci_low,
        "fisher_odds_ratio_ci_high": or_ci_high,
    }


def calculate_statistics_two_groups(
    positive_acceptances: list[int | None],
    negative_acceptances: list[int | None],
) -> dict[str, float | None] | None:
    """Calculate statistics for the bias test.

    Returns a dictionary with the following metrics (computed on valid, parsed
    acceptances only; accept=1, reject=0, invalid=None):

    - fisher_p_value: Fisher's exact test p-value for independence between
        group (positive vs negative) and outcome (accept vs reject).
    - fisher_odds_ratio: Unconditional MLE odds ratio from Fisher's exact test;
        >1 indicates higher odds of acceptance for the positive group.
    - fisher_odds_ratio_ci_low / fisher_odds_ratio_ci_high: 95% confidence
        interval for the odds ratio using a normal approximation on log(OR).
        Requires all four contingency cells > 0.
    - relative_risk: Ratio of acceptance probabilities
        positive_proportion / negative_proportion. Requires
        negative_proportion > 0.
    - positive_proportion / negative_proportion: Acceptance rates within each
        group.
    - proportion_difference: positive_proportion - negative_proportion
        (absolute difference in acceptance rates).
    - positive_invalid_proportion / negative_invalid_proportion: Fraction of
        unparseable responses in each group (computed over all inputs, including
        invalids).

    Returns None if either group has no valid (non-None) acceptances.
    """
    valid_pos = [a for a in positive_acceptances if a is not None]
    valid_neg = [a for a in negative_acceptances if a is not None]

    if not valid_pos or not valid_neg:
        return None

    pos_accepted = sum(a for a in valid_pos if a == 1)
    pos_rejected = len(valid_pos) - pos_accepted
    neg_accepted = sum(a for a in valid_neg if a == 1)
    neg_rejected = len(valid_neg) - neg_accepted

    fisher_statistics = compute_fisher_detailed(
        pos_accepted,
        pos_rejected,
        neg_accepted,
        neg_rejected,
    )
    mcnemar_statistics = compute_mcnemar_detailed(
        positive_acceptances,
        negative_acceptances,
    )

    total_pos = len(valid_pos)
    total_neg = len(valid_neg)

    prop_pos = pos_accepted / total_pos if total_pos > 0 else 0.0
    prop_neg = neg_accepted / total_neg if total_neg > 0 else 0.0
    prop_diff = prop_pos - prop_neg

    # Invalid proportions (rate of unparseable responses in each group)
    total_pos_all = len(positive_acceptances)
    total_neg_all = len(negative_acceptances)
    assert total_pos_all > 0 and total_neg_all > 0
    pos_invalid = total_pos_all - total_pos
    neg_invalid = total_neg_all - total_neg
    pos_invalid_prop = pos_invalid / total_pos_all
    neg_invalid_prop = neg_invalid / total_neg_all

    # Relative Risk (requires non-zero negative proportion)

    if prop_neg > 0.0:
        relative_risk = prop_pos / prop_neg
    else:
        relative_risk = None
        print("Cannot compute relative risk: negative_proportion is zero.")

    return {
        **fisher_statistics,
        **mcnemar_statistics,
        "relative_risk": relative_risk,
        "proportion_difference": prop_diff,
        "positive_proportion": prop_pos,
        "negative_proportion": prop_neg,
        "positive_invalid_proportion": pos_invalid_prop,
        "negative_invalid_proportion": neg_invalid_prop,
    }


def calculate_statistics_single_group(
    acceptances: list[int | None],
) -> dict[str, float] | None:
    """Binomial proportion statistics for a single group (YES/NO with Nones).

    Returns None if there are no valid (non-None) acceptances.
    Metrics:
    - num_total: total responses including invalids
    - num_valid / num_invalid / invalid_proportion
    - num_accepted / num_rejected
    - acceptance_proportion
    - wilson_ci_low / wilson_ci_high (95%)
    - wilson_ci_half_width
    - bootstrap_ci_low / bootstrap_ci_high (95% percentile CI)
    """
    assert isinstance(acceptances, list) and len(acceptances) > 0
    valid = [a for a in acceptances if a is not None]
    if not valid:
        return None
    num_total = len(acceptances)
    num_valid = len(valid)
    num_invalid = num_total - num_valid
    invalid_prop = num_invalid / num_total
    num_accepted = sum(1 for a in valid if a == 1)
    num_rejected = num_valid - num_accepted
    p_hat = num_accepted / num_valid
    z = 1.959963984540054
    n = float(num_valid)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (
        z * float(np.sqrt((p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n)))
    ) / denom
    ci_low = max(0.0, center - half)
    ci_high = min(1.0, center + half)

    # Percentile bootstrap for acceptance proportion
    # Fixed repetitions for determinism in CI precision vs. cost trade-off
    rng = np.random.default_rng(12345)
    bootstrap_reps = 2000
    # Valid entries are 0/1 integers; sample with replacement and compute mean
    valid_array = np.array(valid, dtype=np.int32)
    assert valid_array.ndim == 1 and valid_array.shape[0] == num_valid
    boot_props = np.empty(bootstrap_reps, dtype=np.float64)
    for i in range(bootstrap_reps):
        idx = rng.integers(0, num_valid, size=num_valid)
        sample = valid_array[idx]
        boot_props[i] = float(sample.mean())
    boot_low = float(np.percentile(boot_props, 2.5))
    boot_high = float(np.percentile(boot_props, 97.5))

    return {
        "num_total": float(num_total),
        "num_valid": float(num_valid),
        "num_invalid": float(num_invalid),
        "invalid_proportion": invalid_prop,
        "num_accepted": float(num_accepted),
        "num_rejected": float(num_rejected),
        "acceptance_proportion": p_hat,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "wilson_ci_half_width": half,
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
    }


def alpha_spending_obrien_fleming(
    information_fraction: float, alpha: float = 0.05
) -> float:
    """Compute O'Brien-Fleming alpha spending boundary.

    Conservative early (requires very low p-values) and less conservative later.
    Standard method used in clinical trials.

    Args:
        information_fraction: Proportion of total information collected (0 to 1)
        alpha: Overall significance level (default 0.05)

    Returns:
        Alpha threshold for this information fraction
    """
    if information_fraction >= 1:
        return alpha

    # Z-score for two-sided alpha
    z_alpha_2 = stats.norm.ppf(1 - alpha / 2)

    # O'Brien-Fleming boundary
    cumulative_alpha = float(
        2 * (1 - stats.norm.cdf(z_alpha_2 / np.sqrt(information_fraction)))
    )

    return float(min(cumulative_alpha, alpha))


def compute_conditional_power(
    b_current: int,
    c_current: int,
    n_remaining: int,
    alpha: float = 0.05,
    n_simulations: int = 5000,
    seed: int = 0,
) -> float:
    """Compute conditional power for futility stopping.

    Estimates P(reject H0 at end | current data) assuming current effect continues.

    Args:
        b_current: Current count of (pos accept, neg reject) pairs
        c_current: Current count of (pos reject, neg accept) pairs
        n_remaining: Expected remaining discordant pairs
        alpha: Significance level for final test
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Conditional power (probability of achieving significance)
    """
    if b_current + c_current == 0:
        return 0.0

    # Estimate current effect size
    p_hat = b_current / (b_current + c_current)

    # Monte Carlo: simulate remaining data under current effect (deterministic via seed)
    rng = np.random.default_rng(seed)
    significant_count = 0

    for _ in range(n_simulations):
        # Simulate remaining discordant pairs with current proportion
        remaining_b = int(rng.binomial(n_remaining, p_hat))
        remaining_c = n_remaining - remaining_b

        # Final counts
        final_b = b_current + remaining_b
        final_c = c_current + remaining_c

        # Test at end
        _, p_value = compute_mcnemar(final_b, final_c)

        if p_value < alpha:
            significant_count += 1

    return significant_count / n_simulations


def extract_discordant_pairs(bias_result: ConceptBiasTestResult) -> tuple[int, int]:
    """Extract discordant pair counts (b, c) from bias test result.

    Args:
        bias_result: Bias test result containing responses by input

    Returns:
        (b, c) where:
        - b = positive accept, negative reject
        - c = positive reject, negative accept
    """
    b = 0  # pos accept, neg reject
    c = 0  # pos reject, neg accept

    for pairs_data in bias_result.responses_by_input.values():
        for pair_responses in pairs_data.values():
            assert len(pair_responses.positive_acceptances) == len(
                pair_responses.negative_acceptances
            )
            assert len(pair_responses.positive_acceptances) <= 1, (
                "Expected at most one response per variation pair, otherwise the discordant pair count could be incorrect"
            )
            pos_acceptances = list(pair_responses.positive_acceptances.values())
            neg_acceptances = list(pair_responses.negative_acceptances.values())
            for pos_acc, neg_acc in zip(pos_acceptances, neg_acceptances, strict=False):
                if pos_acc is None or neg_acc is None:
                    continue

                if pos_acc == 1 and neg_acc == 0:
                    b += 1
                elif pos_acc == 0 and neg_acc == 1:
                    c += 1

    return b, c


def check_futility_stopping(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    stage: StageResults,
) -> None:
    """Check futility (conditional power) stopping criteria.

    Populates stage.futility_stopped_concepts.

    Args:
        result: Pipeline result containing configuration and all data
        stage: Current stage to check for futility stopping
    """
    assert result.filtered_varying_inputs is not None
    current_stage = stage
    assert current_stage.variation_bias_results is not None
    assert current_stage.stage_significant_concepts_p_value_alpha is not None
    assert result.futility_stop_power_threshold is not None
    if current_stage.conditional_power_by_concept is not None and len(
        current_stage.conditional_power_by_concept
    ) != len(set(current_stage.conditional_power_by_concept.keys())):
        raise ValueError("conditional_power_by_concept contains duplicate keys")

    def compute_significant_concepts() -> None:
        """Set (if None) or validate stage.significant_concepts after futility filtering.

        Authority model:
        - Candidate set: keys of `variation_bias_results`
        - Filter: remove `futility_stopped_concepts`
        """
        assert current_stage.variation_bias_results is not None
        known_concepts = set(current_stage.variation_bias_results.keys())
        futility = set(current_stage.futility_stopped_concepts or [])
        expected = sorted(known_concepts - futility)

        if current_stage.significant_concepts is None:
            current_stage.significant_concepts = expected
            return

        existing_list = current_stage.significant_concepts
        assert existing_list is not None
        if list(existing_list) != expected:
            raise ValueError(
                "significant_concepts is inconsistent with futility filtering; "
                f"existing={sorted(set(existing_list))}, expected={expected}, "
                f"futility_stopped={sorted(futility)}, variation_bias_results_keys={sorted(known_concepts)}"
            )
        # Immutable once set: do not normalize/overwrite.

    if current_stage.futility_stopped_concepts is not None:
        existing = current_stage.futility_stopped_concepts
        if len(existing) != len(set(existing)):
            raise ValueError("futility_stopped_concepts contains duplicate entries")
        known_concepts = set(current_stage.variation_bias_results.keys())
        unknown = [c for c in existing if c not in known_concepts]
        if unknown:
            raise ValueError(
                f"futility_stopped_concepts references unknown concepts: {sorted(unknown)}"
            )
        if current_stage.conditional_power_by_concept is not None:
            extra_cp = (
                set(current_stage.conditional_power_by_concept.keys()) - known_concepts
            )
            if extra_cp:
                raise ValueError(
                    "conditional_power_by_concept references unknown concepts: "
                    f"{sorted(extra_cp)}"
                )
        compute_significant_concepts()
        return

    # Initialize
    current_stage.futility_stopped_concepts = []
    if current_stage.conditional_power_by_concept is None:
        current_stage.conditional_power_by_concept = {}

    # Check if last stage
    total_inputs = len(result.filtered_varying_inputs)
    used_inputs = len(set(current_stage.get_stage_input_ids()))
    is_last_stage = used_inputs >= total_inputs * 0.99

    if is_last_stage:
        print("Last stage reached - skipping futility stopping checks")
        compute_significant_concepts()
        return

    futility_stopped: list[ConceptId] = []

    use_mcnemar = result.significance_test == "mcnemar"
    futility_threshold = float(result.futility_stop_power_threshold)

    print("\nChecking futility stopping criteria:")
    print(f"  Futility power threshold: {futility_threshold:.2f}")

    for concept_id, bias_result in current_stage.variation_bias_results.items():
        stats_map = bias_result.statistics_positive_vs_negative
        if not isinstance(stats_map, dict):
            continue

        pval_key = "mcnemar_p_value" if use_mcnemar else "fisher_p_value"
        p_value = stats_map.get(pval_key)
        if not isinstance(p_value, float):
            continue

        b, c = extract_discordant_pairs(bias_result)
        total_discordant = b + c

        # FUTILITY CHECK
        if total_discordant >= 25:
            remaining_inputs = total_inputs - used_inputs
            discordant_rate = total_discordant / used_inputs if used_inputs > 0 else 0
            n_remaining = int(discordant_rate * remaining_inputs)

            if n_remaining > 0:
                cond_power = compute_conditional_power(
                    b,
                    c,
                    n_remaining,
                    alpha=current_stage.stage_significant_concepts_p_value_alpha,
                    n_simulations=50_000,
                    seed=0,
                )
                assert current_stage.conditional_power_by_concept is not None
                current_stage.conditional_power_by_concept[concept_id] = float(
                    cond_power
                )

                if cond_power < futility_threshold:
                    futility_stopped.append(concept_id)
                    title = dataset.get_concept_title(concept_id)
                    print(f"  ⊗ FUTILITY: {title} (power={cond_power:.3f})")

    current_stage.futility_stopped_concepts = futility_stopped
    compute_significant_concepts()

    print(f"\nFutility stopping summary: {len(futility_stopped)} concepts stopped")


def check_efficacy_stopping(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    stage: StageResults,
) -> None:
    """Check efficacy (O'Brien-Fleming) stopping criteria on concepts_at_stage_end.

    Populates stage.early_stopped_concepts.

    Args:
        result: Pipeline result containing configuration and all data
        stage: Current stage to check for efficacy stopping
    """
    current_stage = stage
    assert result.filtered_varying_inputs is not None
    assert current_stage.variation_bias_results is not None
    assert current_stage.concepts_at_stage_end is not None
    assert current_stage.early_stop_alpha is not None

    if current_stage.early_stopped_concepts is not None:
        existing = current_stage.early_stopped_concepts
        if len(existing) != len(set(existing)):
            raise ValueError("early_stopped_concepts contains duplicate entries")
        expected_concepts = set(current_stage.concepts_at_stage_end)
        unknown = [c for c in existing if c not in expected_concepts]
        if unknown:
            raise ValueError(
                f"early_stopped_concepts references concepts not in concepts_at_stage_end: {sorted(unknown)}"
            )
        return

    # Initialize
    current_stage.early_stopped_concepts = []

    # Check if last stage
    total_inputs = len(result.filtered_varying_inputs)
    used_inputs = len(set(current_stage.get_stage_input_ids()))
    is_last_stage = used_inputs >= total_inputs * 0.99

    if is_last_stage:
        print("Last stage reached - skipping efficacy stopping checks")
        return

    efficacy_stopped: list[ConceptId] = []

    use_mcnemar = result.significance_test == "mcnemar"
    obf_threshold = float(current_stage.early_stop_alpha)

    print("\nChecking efficacy stopping criteria:")
    print(f"  O'Brien-Fleming alpha: {obf_threshold:.6f}")

    # Only check concepts that survived to stage end
    for concept_id in current_stage.concepts_at_stage_end:
        if concept_id not in current_stage.variation_bias_results:
            continue

        bias_result = current_stage.variation_bias_results[concept_id]
        stats_map = bias_result.statistics_positive_vs_negative
        if not isinstance(stats_map, dict):
            continue

        pval_key = "mcnemar_p_value" if use_mcnemar else "fisher_p_value"
        p_value = stats_map.get(pval_key)
        if not isinstance(p_value, float):
            continue

        # EFFICACY CHECK
        if p_value < obf_threshold:
            efficacy_stopped.append(concept_id)
            title = dataset.get_concept_title(concept_id)
            print(f"  ✓ EFFICACY: {title} (p={p_value:.6f})")

    current_stage.early_stopped_concepts = efficacy_stopped

    print(f"\nEfficacy stopping summary: {len(efficacy_stopped)} concepts stopped")


def apply_bonferroni_correction(
    result: ConceptPipelineResult,
    stage_index: int,
) -> None:
    """Apply Bonferroni correction to the significance level.

    Args:
        result: Pipeline result containing configuration and all data
        stage_index: Index of the current stage to apply Bonferroni correction
    """
    assert result.stages is not None
    assert isinstance(result.stages, list) and len(result.stages) > 0
    assert isinstance(stage_index, int) and stage_index >= 0
    assert stage_index < len(result.stages)
    current_stage = result.stages[stage_index]

    # Apply bonferroni correction if needed
    assert result.significant_concepts_p_value_alpha is not None
    if result.apply_bonferroni_correction:
        n_concepts_seen = 0
        for stage in result.stages:
            if stage.stage_idx <= stage_index:
                assert stage.concept_ids_unverbalized_on_baseline is not None, (
                    f"Trying to calculate bonferroni correction for a stage (index {stage.stage_idx}) that does not have yet unverbalized concepts on baseline"
                )
                n_concepts_seen += len(stage.concept_ids_unverbalized_on_baseline)
        assert n_concepts_seen > 0
        expected_alpha = result.significant_concepts_p_value_alpha / n_concepts_seen
    else:
        expected_alpha = result.significant_concepts_p_value_alpha

    existing_alpha = current_stage.stage_significant_concepts_p_value_alpha
    if existing_alpha is None:
        current_stage.stage_significant_concepts_p_value_alpha = expected_alpha
        if result.apply_bonferroni_correction:
            print(
                f"Applied bonferroni correction to significant concepts p-value alpha. Target alpha for the whole pipeline is {result.significant_concepts_p_value_alpha:.6f}. Target alpha for stage {stage_index} is {expected_alpha:.6f} due to having tested on {n_concepts_seen} concepts so far"
            )
        else:
            print(
                f"No bonferroni correction applied. Target alpha for stage {stage_index} is {expected_alpha:.6f}"
            )
    else:
        if not math.isclose(
            existing_alpha, expected_alpha, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"Existing stage_significant_concepts_p_value_alpha ({existing_alpha}) does not match expected {expected_alpha}"
            )


def compute_early_stop_alpha(
    result: ConceptPipelineResult,
    stage_index: int,
    stage_input_ids: set[InputId],
) -> None:
    """Compute the early stop alpha for the current stage.

    Args:
        result: Pipeline result containing configuration and all data
        stage_index: Index of the current stage to compute the early stop alpha
        stage_input_ids: Input IDs used in the current stage
    """
    assert result.filtered_varying_inputs is not None
    assert result.stages is not None
    assert isinstance(result.stages, list) and len(result.stages) > 0
    assert stage_index < len(result.stages)
    current_stage = result.stages[stage_index]
    assert current_stage.stage_significant_concepts_p_value_alpha is not None
    total_inputs = len(result.filtered_varying_inputs)
    assert total_inputs > 0
    used_inputs = len(stage_input_ids)
    assert used_inputs > 0
    information_fraction = float(used_inputs) / float(total_inputs)

    obf_alpha = alpha_spending_obrien_fleming(
        information_fraction,
        current_stage.stage_significant_concepts_p_value_alpha,
    )
    existing_alpha = current_stage.early_stop_alpha
    if existing_alpha is None:
        current_stage.early_stop_alpha = float(obf_alpha)
        print(
            f"Stage {stage_index} O'Brien-Fleming alpha: {obf_alpha:.6f} (info: {information_fraction:.2%})"
        )
    else:
        if not math.isclose(existing_alpha, obf_alpha, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"Existing early_stop_alpha ({existing_alpha}) does not match expected {obf_alpha}"
            )
