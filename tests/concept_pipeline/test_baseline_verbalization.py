from pathlib import Path

import pytest

from biases_in_the_blind_spot.concept_pipeline.baseline_verbalization import (
    analyze_verbalization_on_baseline_for_stage,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from tests.concept_pipeline.helpers import (
    NoCallBaselineDetector,
    StubBaselineDetector,
    baseline_flags,
    make_dataset,
    make_stage_baseline,
)

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.anyio
async def test_baseline_existing_data_no_top_up(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    dataset = make_dataset(cid, iid)
    existing_flags = {cid: {iid: {rid: baseline_flags(verbalized=True, witness="w")}}}
    stage = make_stage_baseline(cid, iid, baseline_map=existing_flags, unverbalized=[])
    result = ConceptPipelineResult(
        baseline_responses_by_input={iid: {rid: "resp"}},
        baseline_verbalization_threshold=0.25,
        stages=[stage],
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.plot_concept_baseline_verbalization",
        lambda *args, **kwargs: None,
    )

    detector = NoCallBaselineDetector()
    assert result.stages is not None
    before = result.stages[0].concept_verbalization_on_baseline_responses
    await analyze_verbalization_on_baseline_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        baseline_verbalization_threshold=0.25,
        verbalization_detector=detector,
    )
    assert result.stages is not None
    assert result.stages[0].concept_verbalization_on_baseline_responses == before, (
        "Stored verbalization data must remain unchanged"
    )
    assert stage.concept_ids_unverbalized_on_baseline == []


@pytest.mark.anyio
async def test_baseline_populates_missing(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    dataset = make_dataset(cid, iid)
    stage = make_stage_baseline(cid, iid, baseline_map=None, unverbalized=None)
    result = ConceptPipelineResult(
        baseline_responses_by_input={iid: {rid: "resp"}},
        baseline_verbalization_threshold=0.5,
        stages=[stage],
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.plot_concept_baseline_verbalization",
        lambda *args, **kwargs: None,
    )

    stub_detector = StubBaselineDetector(
        {cid: {iid: {rid: baseline_flags(verbalized=False, witness="")}}}
    )
    await analyze_verbalization_on_baseline_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        baseline_verbalization_threshold=0.5,
        verbalization_detector=stub_detector,
    )

    assert stub_detector.called
    filled = stage.concept_verbalization_on_baseline_responses
    assert filled is not None
    assert filled[cid][iid][rid].verbalized is False
    assert stage.concept_ids_unverbalized_on_baseline == [cid]
