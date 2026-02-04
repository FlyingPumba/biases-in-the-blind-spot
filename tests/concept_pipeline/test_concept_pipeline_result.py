from pathlib import Path

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId


def test_get_figures_root_and_stage_root(tmp_path: Path):
    result = ConceptPipelineResult(experiment_key="exp1")
    root = result.get_figures_root(tmp_path)
    assert root.exists() and root.name == "exp1"
    stage = StageResults(
        stage_idx=2,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [InputId()]},
        concepts_at_stage_start=[ConceptId()],
    )
    stage_root = result.get_stage_figures_root(tmp_path, stage)
    assert stage_root.exists()
    assert stage_root.name == "stage-2"
