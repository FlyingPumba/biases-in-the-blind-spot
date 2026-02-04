from biases_in_the_blind_spot.concept_pipeline.statistics import (
    alpha_spending_obrien_fleming,
    calculate_statistics_single_group,
    calculate_statistics_two_groups,
    compute_fisher_detailed,
    compute_mcnemar,
    compute_mcnemar_detailed,
)


def test_compute_mcnemar_and_detailed():
    stat, p = compute_mcnemar(3, 1)
    assert stat > 0
    assert 0 <= p <= 1
    details = compute_mcnemar_detailed([1, 1, 0, 0], [0, 1, 1, 0])
    assert details["mcnemar_total_pairs"] == 4.0
    assert details["mcnemar_pos_accept_neg_reject"] == 1.0
    assert details["mcnemar_pos_reject_neg_accept"] == 1.0


def test_compute_fisher_detailed():
    out = compute_fisher_detailed(5, 1, 1, 5)
    p_val = out["fisher_p_value"]
    odds = out["fisher_odds_ratio"]
    assert p_val is not None and 0 <= p_val <= 1
    assert odds is not None and odds > 0


def test_calculate_statistics_two_groups_valid():
    pos: list[int | None] = [1, 1, 0, 1]
    neg: list[int | None] = [0, 0, 1, 0]
    out = calculate_statistics_two_groups(pos, neg)
    assert out is not None
    assert out["positive_proportion"] == 0.75
    assert out["negative_proportion"] == 0.25
    assert out["proportion_difference"] == 0.5


def test_calculate_statistics_two_groups_invalid_returns_none():
    assert calculate_statistics_two_groups([None], [0]) is None
    assert calculate_statistics_two_groups([1], [None]) is None


def test_calculate_statistics_single_group_basic():
    out = calculate_statistics_single_group([1, 0, 1, None])
    assert out is not None
    assert out["num_valid"] == 3.0
    assert out["num_invalid"] == 1.0
    assert out["acceptance_proportion"] == 2 / 3
    assert 0 <= out["wilson_ci_low"] <= out["wilson_ci_high"] <= 1


def test_calculate_statistics_single_group_no_valid():
    assert calculate_statistics_single_group([None, None]) is None


def test_alpha_spending_obrien_fleming():
    early = alpha_spending_obrien_fleming(0.1, alpha=0.05)
    late = alpha_spending_obrien_fleming(0.9, alpha=0.05)
    assert early < late <= 0.05
