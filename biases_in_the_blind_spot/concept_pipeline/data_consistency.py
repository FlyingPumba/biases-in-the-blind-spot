from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)


def validate_concepts_at_stage_start(
    dataset: ConceptPipelineDataset,
    result: ConceptPipelineResult,
    current_stage: StageResults,
) -> None:
    """Check that concepts_at_stage_start matches what the pipeline expects; never mutates."""
    assert result.stages is not None

    stage_index = current_stage.stage_idx
    if stage_index == 0:
        pipeline_concepts = dataset.get_pipeline_concepts()
        expected = [c.id for c in pipeline_concepts]
    else:
        assert len(result.stages) > 0
        prev = result.stages[stage_index - 1]
        assert prev.concepts_at_stage_end is not None
        expected = list(prev.concepts_at_stage_end)
        excluded = set(prev.early_stopped_concepts or []) | set(
            prev.futility_stopped_concepts or []
        )
        if excluded:
            expected = [c for c in expected if c not in excluded]

    if current_stage.concepts_at_stage_start is None:
        raise ValueError(
            f"Stage {stage_index}: concepts_at_stage_start is missing; expected {len(expected)} concepts"
        )

    actual = list(current_stage.concepts_at_stage_start)
    extra = set(actual) - set(expected)
    missing = set(expected) - set(actual)
    if extra or missing:
        raise ValueError(
            f"Stage {stage_index}: concepts_at_stage_start mismatch; "
            f"extra={sorted(extra)} missing={sorted(missing)}"
        )
