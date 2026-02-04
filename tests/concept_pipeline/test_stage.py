import pytest

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.stage import (
    compute_concepts_in_last_stage,
    start_new_stage,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)


class _FakeDataset:
    def __init__(self, dedup_concepts, select_outputs):
        self.deduplicated_concepts = dedup_concepts
        self.final_concepts = dedup_concepts
        self.input_clusters = object()
        self._select_outputs = select_outputs

    def get_pipeline_concepts(self):
        return self.final_concepts

    def select_inputs_from_each_representative_cluster(
        self, _k, _output_dir, whitelisted_input_ids
    ):
        return self._select_outputs


class _Concept:
    def __init__(self, cid):
        self.id = cid


def test_compute_concepts_in_last_stage_filters_early_stopped():
    cid1 = ConceptId()
    cid2 = ConceptId()
    stage0 = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[cid1, cid2],
        concepts_at_stage_end=[cid1, cid2],
        early_stopped_concepts=[cid2],
    )
    result = ConceptPipelineResult(stages=[stage0])
    dataset = _FakeDataset(
        dedup_concepts=[_Concept(cid1), _Concept(cid2)], select_outputs={}
    )
    out = compute_concepts_in_last_stage(result, 1, dataset=dataset)  # type: ignore
    assert out == [cid1], "Early-stopped concepts must be excluded"


def test_start_new_stage_copies_previous_stage_data():
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    pid = VariationPairId()
    prev_stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        concepts_at_stage_end=[cid],
        concept_verbalization_on_baseline_responses={
            cid: {iid: {rid: VerbalizationCheckResult(True, "w")}}
        },
        variation_bias_results={},
        concept_verbalization_on_variation_responses={
            cid: {iid: {pid: VariationPairVerbalization({}, {})}}
        },
    )
    result = ConceptPipelineResult(
        stages=[prev_stage],
        filtered_varying_inputs=[iid],
    )
    select_outputs = {0: [iid]}
    dataset = _FakeDataset(
        dedup_concepts=[_Concept(cid)], select_outputs=select_outputs
    )

    def k_fn(_idx: int) -> int:
        return 1

    new_stage = start_new_stage(
        result,
        1,
        dataset=dataset,  # type: ignore
        representative_inputs_k_per_stage_index_fn=k_fn,
        output_dir=None,  # type: ignore
    )
    assert new_stage.stage_idx == 1
    assert result.stages[-1] is new_stage  # type: ignore
    assert new_stage.concept_verbalization_on_baseline_responses is None
    assert new_stage.concept_verbalization_on_variation_responses is None
    assert new_stage.variation_bias_results is None


def test_start_new_stage_reuses_existing_stage_when_matching():
    cid = ConceptId()
    iid = InputId()
    stage_existing = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
    )
    result = ConceptPipelineResult(
        stages=[stage_existing],
        filtered_varying_inputs=[iid],
    )
    dataset = _FakeDataset(dedup_concepts=[_Concept(cid)], select_outputs={0: [iid]})

    def k_fn(_idx: int) -> int:
        return 1

    out = start_new_stage(
        result,
        0,
        dataset=dataset,  # type: ignore
        representative_inputs_k_per_stage_index_fn=k_fn,
        output_dir=None,  # type: ignore
    )
    assert out is stage_existing

    with pytest.raises(ValueError):
        start_new_stage(
            result,
            0,
            dataset=_FakeDataset(  # pyright: ignore[reportArgumentType]
                dedup_concepts=[_Concept(cid)], select_outputs={1: [iid]}
            ),
            representative_inputs_k_per_stage_index_fn=k_fn,
            output_dir=None,  # type: ignore
        )
