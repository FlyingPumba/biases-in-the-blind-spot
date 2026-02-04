import pytest

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.data_consistency import (
    validate_concepts_at_stage_start,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from tests.concept_pipeline.helpers import make_concept


def _dataset_with_concepts(cids):
    concepts = [make_concept(cid) for cid in cids]
    return ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{vary}",
        input_parameters={"vary": "x"},
        varying_input_param_name="vary",
        varying_inputs={InputId(): "text"},
        sanitized_varying_inputs={InputId(): "text"},
        concepts=concepts,
        deduplicated_concepts=concepts,
        final_concepts=concepts,
    )


def test_validate_stage0_matches_dataset():
    cid = ConceptId()
    dataset = _dataset_with_concepts([cid])
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid],
    )
    result = ConceptPipelineResult(stages=[stage])
    validate_concepts_at_stage_start(dataset, result, stage)


def test_validate_stage0_raises_on_mismatch():
    cid = ConceptId()
    dataset = _dataset_with_concepts([cid])
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[],
    )
    result = ConceptPipelineResult(stages=[stage])
    with pytest.raises(ValueError):
        validate_concepts_at_stage_start(dataset, result, stage)


def test_validate_stage1_uses_prev_end_and_filters_early_stopped():
    cid1 = ConceptId()
    cid2 = ConceptId()
    dataset = _dataset_with_concepts([cid1, cid2])
    prev = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid1, cid2],
        concepts_at_stage_end=[cid1, cid2],
        early_stopped_concepts=[cid2],
    )
    curr = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid1],
    )
    result = ConceptPipelineResult(stages=[prev, curr])
    validate_concepts_at_stage_start(dataset, result, curr)


def test_validate_stage1_missing_prev_end_raises():
    cid = ConceptId()
    dataset = _dataset_with_concepts([cid])
    prev = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid],
    )
    curr = StageResults(
        stage_idx=1,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid],
    )
    result = ConceptPipelineResult(stages=[prev, curr])
    with pytest.raises(AssertionError):
        validate_concepts_at_stage_start(dataset, result, curr)
