from pathlib import Path

# isort: skip_file

import pytest

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.variation_verbalization import (
    analyze_verbalization_on_variations_for_stage,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)

from tests.concept_pipeline.helpers import (
    NoCallVariationDetector,
    StubVariationDetector,
    make_bias_result,
    make_dataset,
    make_stage_variation,
)

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.anyio
async def test_variation_existing_data_no_top_up(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()

    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    bias_res = make_bias_result(iid, pid, responses)
    stored = VariationPairVerbalization(
        positive_variation_responses_verbalizations={
            rid_pos: VerbalizationCheckResult(True, "w")
        },
        negative_variation_responses_verbalizations={
            rid_neg: VerbalizationCheckResult(False, "")
        },
    )
    stage = make_stage_variation(
        cid,
        iid,
        var_map={cid: bias_res},
    )
    stage.concept_verbalization_on_variation_responses = {cid: {iid: {pid: stored}}}
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    variation_pair = VariationPair("p", "n")
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=variation_pair
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    detector = NoCallVariationDetector()
    before = stage.concept_verbalization_on_variation_responses
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        verbalization_detector=detector,
    )
    assert stage.concept_verbalization_on_variation_responses == before


@pytest.mark.anyio
async def test_variation_populates_missing(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()
    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    bias_res = make_bias_result(iid, pid, responses)
    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.concept_verbalization_on_variation_responses = None
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    variation_pair = VariationPair("p", "n")
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=variation_pair
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    stub_detector = StubVariationDetector(
        {
            cid: {
                iid: {
                    pid: {
                        "positive": {rid_pos: VerbalizationCheckResult(True, "w+")},
                        "negative": {rid_neg: VerbalizationCheckResult(False, "")},
                    }
                }
            }
        }
    )
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        verbalization_detector=stub_detector,
    )

    assert stub_detector.called
    stored = stage.concept_verbalization_on_variation_responses
    assert stored is not None
    pair_data = stored[cid][iid][pid]
    assert pair_data.positive_variation_responses_verbalizations[rid_pos].verbalized
    assert (
        pair_data.negative_variation_responses_verbalizations[rid_neg].verbalized
        is False
    )


@pytest.mark.anyio
async def test_variation_raises_on_missing_positive_ids(monkeypatch):
    cid = ConceptId("8b031a46-816c-463d-b735-83639c3ee686")
    iid = InputId("001df7a5-de16-445e-af13-1e3a6fb3b745")
    pid = VariationPairId("41a6c28e-e03d-46ab-8cf9-2e08dc4f7a84")
    rid_pos = ResponseId("e30396ce-c046-4151-ba3d-d5f788096573")
    rid_neg = ResponseId("91ef94fa-80cf-48a9-8711-3314383d3170")

    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    bias_res = make_bias_result(iid, pid, responses)
    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.concept_verbalization_on_variation_responses = None
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    variation_pair = VariationPair("p", "n")
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=variation_pair
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    stub_detector = StubVariationDetector(
        {
            cid: {
                iid: {
                    pid: {
                        # Detector drops the positive response id entirely
                        "positive": {},
                        "negative": {rid_neg: VerbalizationCheckResult(True, "w-")},
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Positive response ids mismatch"):
        await analyze_verbalization_on_variations_for_stage(
            result,
            stage,
            dataset=dataset,
            output_dir=Path("/tmp"),
            verbalization_detector=stub_detector,
        )


@pytest.mark.anyio
async def test_variation_skips_zero_flipped_pairs(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()

    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 1},
    )
    bias_res = make_bias_result(iid, pid, responses)
    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.concept_verbalization_on_variation_responses = None
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    variation_pair = VariationPair("p", "n")
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=variation_pair
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    detector = NoCallVariationDetector()
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        verbalization_detector=detector,
    )

    assert stage.concept_verbalization_on_variation_responses == {cid: {}}


@pytest.mark.anyio
async def test_variation_verbalization_raises_on_futility_overlap(monkeypatch):
    """Futility-stopped concepts must never be processed for variation verbalization."""

    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()

    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 0},
    )
    bias_res = make_bias_result(iid, pid, responses)
    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.futility_stopped_concepts = [cid]

    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    variation_pair = VariationPair("p", "n")
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=variation_pair
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    detector = NoCallVariationDetector()
    with pytest.raises(
        AssertionError, match="Futility-stopped concepts must not be processed"
    ):
        await analyze_verbalization_on_variations_for_stage(
            result,
            stage,
            dataset=dataset,
            output_dir=Path("/tmp"),
            verbalization_detector=detector,
        )


@pytest.mark.anyio
async def test_variation_verbalization_only_requests_flipped_pairs(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    pid_flipped = VariationPairId()
    pid_not_flipped = VariationPairId()
    rid_pos_1 = ResponseId()
    rid_neg_1 = ResponseId()
    rid_pos_2 = ResponseId()
    rid_neg_2 = ResponseId()

    flipped = VariationPairResponses(
        positive_responses={rid_pos_1: "resp+1"},
        negative_responses={rid_neg_1: "resp-1"},
        positive_acceptances={rid_pos_1: 1},
        negative_acceptances={rid_neg_1: 0},
    )
    not_flipped = VariationPairResponses(
        positive_responses={rid_pos_2: "resp+2"},
        negative_responses={rid_neg_2: "resp-2"},
        positive_acceptances={rid_pos_2: 1},
        negative_acceptances={rid_neg_2: 1},
    )

    # Custom bias result with two pairs: only one is flipped.
    bias_res = make_bias_result(iid, pid_flipped, flipped)
    bias_res.responses_by_input[iid][pid_not_flipped] = not_flipped
    assert bias_res.flipped_variation_pairs is not None
    bias_res.flipped_variation_pairs[iid][pid_not_flipped] = False

    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.concept_verbalization_on_variation_responses = None
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )

    # Dataset variations must include both pairs for the input.
    dataset = make_dataset(
        cid,
        iid,
        variation_pair_id=pid_flipped,
        variation_pair=VariationPair("p1", "n1"),
    )
    assert dataset.variations is not None
    dataset.variations[cid][iid][pid_not_flipped] = VariationPair("p2", "n2")  # type: ignore[index]

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    class _CapturingDetector(VerbalizationDetector):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def is_verbalized_variations_batch(self, _dataset, responses_to_analyze):
            nested_requests = responses_to_analyze
            self.calls.append(nested_requests)
            # Only the flipped pair should be requested.
            assert set(nested_requests.keys()) == {cid}
            assert set(nested_requests[cid].keys()) == {iid}
            assert set(nested_requests[cid][iid].keys()) == {pid_flipped}
            # Return a valid response for the flipped pair.
            return {
                cid: {
                    iid: {
                        pid_flipped: {
                            "positive": {
                                rid_pos_1: VerbalizationCheckResult(True, "w")
                            },
                            "negative": {
                                rid_neg_1: VerbalizationCheckResult(False, "")
                            },
                        }
                    }
                }
            }

    detector = _CapturingDetector()
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        verbalization_detector=detector,
    )

    assert len(detector.calls) == 1
    stored = stage.concept_verbalization_on_variation_responses
    assert stored is not None
    assert set(stored[cid][iid].keys()) == {pid_flipped}


@pytest.mark.anyio
async def test_variation_verbalization_skips_zero_flipped_pairs(monkeypatch):
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rid_pos = ResponseId()
    rid_neg = ResponseId()

    # Non-flipped pair: acceptances equal.
    responses = VariationPairResponses(
        positive_responses={rid_pos: "resp+"},
        negative_responses={rid_neg: "resp-"},
        positive_acceptances={rid_pos: 1},
        negative_acceptances={rid_neg: 1},
    )
    bias_res = make_bias_result(iid, pid, responses)
    assert bias_res.flipped_variation_pairs is not None
    assert bias_res.flipped_variation_pairs[iid][pid] is False

    stage = make_stage_variation(cid, iid, var_map={cid: bias_res})
    stage.concept_verbalization_on_variation_responses = None
    result = ConceptPipelineResult(
        variations_verbalization_threshold=0.5,
        stages=[stage],
    )
    dataset = make_dataset(
        cid, iid, variation_pair_id=pid, variation_pair=VariationPair("p", "n")
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        lambda *args, **kwargs: None,
    )

    detector = NoCallVariationDetector()
    await analyze_verbalization_on_variations_for_stage(
        result,
        stage,
        dataset=dataset,
        output_dir=Path("/tmp"),
        verbalization_detector=detector,
    )

    assert stage.concept_verbalization_on_variation_responses == {cid: {}}
