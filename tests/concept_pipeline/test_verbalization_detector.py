import pytest

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)
from tests.concept_pipeline.helpers import StubLLMDetector, make_dataset

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.anyio
async def test_is_verbalized_baseline_batch_parses_and_aligns():
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    dataset = make_dataset(cid, iid)

    detector = StubLLMDetector("<verbalized>YES</verbalized><witness>snippet</witness>")
    out = await detector.is_verbalized_baseline_batch(
        dataset, {cid: {iid: {rid: "resp"}}}
    )

    assert cid in out and iid in out[cid] and rid in out[cid][iid]
    result_obj = out[cid][iid][rid]
    assert isinstance(result_obj, VerbalizationCheckResult)
    assert result_obj.verbalized is True
    assert detector.calls, "LLM batch helper should be invoked"


@pytest.mark.anyio
async def test_is_verbalized_variations_batch_parses_and_aligns():
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = make_dataset(cid, iid)
    detector = StubLLMDetector("<verbalized>YES</verbalized><witness>snippet</witness>")

    out = await detector.is_verbalized_variations_batch(
        dataset,
        {
            cid: {
                iid: {
                    pid: {
                        "positive": {ResponseId(): "a"},
                        "negative": {ResponseId(): "b"},
                    }
                }
            }
        },
    )

    assert cid in out and iid in out[cid] and pid in out[cid][iid]
    pair = out[cid][iid][pid]
    assert "positive" in pair and "negative" in pair
    assert all(
        isinstance(v, VerbalizationCheckResult) for v in pair["positive"].values()
    )
    assert all(
        isinstance(v, VerbalizationCheckResult) for v in pair["negative"].values()
    )
    assert detector.calls, "LLM batch helper should be invoked"


@pytest.mark.anyio
async def test_is_verbalized_variations_batch_raises_on_missing_ids():
    # Reproduce the mismatch seen in logs: detector returns missing response ids for a pair
    cid = ConceptId("8b031a46-816c-463d-b735-83639c3ee686")
    iid = InputId("001df7a5-de16-445e-af13-1e3a6fb3b745")
    pid = VariationPairId("41a6c28e-e03d-46ab-8cf9-2e08dc4f7a84")
    pos_id = ResponseId("e30396ce-c046-4151-ba3d-d5f788096573")
    neg_id = ResponseId("91ef94fa-80cf-48a9-8711-3314383d3170")
    dataset = make_dataset(cid, iid)

    class _Detector(VerbalizationDetector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def _generate_batch_llm_response(self, key_to_messages):
            self.calls += 1
            # Simulate detector dropping the positive response id (returns only negative)
            # Key structure: (concept_id, input_idx, pair_idx, side, response_id)
            _, _, _, side, resp_id = list(key_to_messages.keys())[0]
            assert side == "positive"
            assert resp_id == pos_id
            # Return only a negative side entry, missing the expected positive id
            fake_key = (cid, iid, pid, "negative", neg_id)
            return {
                fake_key: "<verbalized>YES</verbalized><witness>dummy</witness>",
            }

    detector = _Detector()

    with pytest.raises(ValueError):
        await detector.is_verbalized_variations_batch(
            dataset,
            {
                cid: {
                    iid: {
                        pid: {
                            "positive": {pos_id: "resp+"},
                            "negative": {neg_id: "resp-"},
                        }
                    }
                }
            },
        )
    assert detector.calls == 1


@pytest.mark.anyio
async def test_is_verbalized_baseline_batch_fails_on_unparsed():
    cid = ConceptId()
    iid = InputId()
    rid = ResponseId()
    dataset = make_dataset(cid, iid)

    class _Detector(VerbalizationDetector):
        async def _generate_batch_llm_response(self, key_to_messages):
            return {list(key_to_messages.keys())[0]: "unparsable"}

    detector = _Detector()
    with pytest.raises(ValueError):
        await detector.is_verbalized_baseline_batch(
            dataset, {cid: {iid: {rid: "resp"}}}
        )


@pytest.mark.anyio
async def test_is_verbalized_variations_batch_fails_on_unparsed():
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    dataset = make_dataset(cid, iid)

    class _Detector(VerbalizationDetector):
        async def _generate_batch_llm_response(self, key_to_messages):
            return {list(key_to_messages.keys())[0]: "unparsable"}

    detector = _Detector()
    with pytest.raises(ValueError):
        await detector.is_verbalized_variations_batch(
            dataset,
            {cid: {iid: {pid: {"positive": {ResponseId(): "a"}, "negative": {}}}}},
        )
