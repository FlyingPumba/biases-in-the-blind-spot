# isort: skip_file

import pytest

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
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
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.statistics import check_futility_stopping
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from tests.concept_pipeline.helpers import make_concept


def test_futility_stopping_is_authority_for_significant_concepts(monkeypatch):
    """Regression: stage.significant_concepts must be reconciled with futility stopping.

    This reproduces the fresh-run failure mode where `variation_bias` set
    `stage.significant_concepts` before futility filtering.
    """
    cid1 = ConceptId()
    cid2 = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()

    # Minimal bias results with required structure.
    vpr = VariationPairResponses(
        positive_responses={rid_pos: "p"},
        negative_responses={rid_neg: "n"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    bias1 = ConceptBiasTestResult(
        responses_by_input={iid: {pid: vpr}},
        statistics_positive_vs_negative={"fisher_p_value": 0.5},
    )
    bias2 = ConceptBiasTestResult(
        responses_by_input={iid: {pid: vpr}},
        statistics_positive_vs_negative={"fisher_p_value": 0.5},
    )

    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=10,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid] * 10},
        concepts_at_stage_start=[cid1, cid2],
        variation_bias_results={cid1: bias1, cid2: bias2},
        stage_significant_concepts_p_value_alpha=0.05,
    )

    dataset = ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{x}",
        input_parameters={},
        varying_input_param_name="x",
        varying_inputs={iid: "A"},
        sanitized_varying_inputs={iid: "A"},
        concepts=[make_concept(cid1), make_concept(cid2)],
        deduplicated_concepts=[make_concept(cid1), make_concept(cid2)],
        variations={
            cid1: {iid: {pid: VariationPair("p", "n")}},
            cid2: {iid: {pid: VariationPair("p", "n")}},
        },
    )

    # Ensure futility logic runs and stops concepts deterministically.
    # - Force discordant pairs >= futility gate (25).
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.statistics.extract_discordant_pairs",
        lambda _bias_result: (25, 0),
    )
    # - Force conditional power below threshold.
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.statistics.compute_conditional_power",
        lambda *args, **kwargs: 0.0,
    )

    result = ConceptPipelineResult(
        stages=[stage],
        filtered_varying_inputs=[InputId() for _ in range(100)],
        futility_stop_power_threshold=0.10,
        significance_test="fisher",
    )

    check_futility_stopping(dataset, result, stage)

    # After futility stopping, significant_concepts must reflect the futility-filtered set.
    assert stage.significant_concepts == []

    # Strictness: pre-setting significant_concepts to an inconsistent value is an error.
    stage.significant_concepts = [cid1, cid2]
    stage.futility_stopped_concepts = [cid1]
    with pytest.raises(
        ValueError, match="significant_concepts is inconsistent with futility filtering"
    ):
        check_futility_stopping(dataset, result, stage)
