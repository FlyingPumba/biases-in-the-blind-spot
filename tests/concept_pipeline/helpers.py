from typing import Any

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    StageResults,
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
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import (
    VerbalizationDetector,
)


# --- Detector stubs ---
class NoCallBaselineDetector(VerbalizationDetector):
    async def is_verbalized_baseline_batch(self, *_args, **_kwargs):
        raise AssertionError("Baseline detector should not be invoked")


class NoCallVariationDetector(VerbalizationDetector):
    async def is_verbalized_variations_batch(self, *_args, **_kwargs):
        raise AssertionError("Variation detector should not be invoked")


class StubBaselineDetector(VerbalizationDetector):
    def __init__(self, responses: Any):
        super().__init__()
        self._responses = responses
        self.called = False

    async def is_verbalized_baseline_batch(self, *_args, **_kwargs):
        self.called = True
        return self._responses


class StubVariationDetector(VerbalizationDetector):
    def __init__(self, responses: Any):
        super().__init__()
        self._responses = responses
        self.called = False

    async def is_verbalized_variations_batch(self, *_args, **_kwargs):
        self.called = True
        return self._responses


class StubLLMDetector(VerbalizationDetector):
    """For detector unit tests: bypass chat_limiter and return fixed XML."""

    def __init__(self, xml: str):
        super().__init__()
        self.calls: list[dict] = []
        self._xml = xml

    async def _generate_batch_llm_response(self, key_to_messages):
        self.calls.append(dict(key_to_messages))
        return dict.fromkeys(key_to_messages, self._xml)


# --- Dataset and stage builders ---
def make_concept(concept_id: ConceptId) -> Concept:
    return Concept(
        title="My Concept—Category",
        verbalization_check_guide="guide",
        removal_action="rem",
        addition_action="add",
        id=concept_id,
    )


def make_dataset(
    concept_id: ConceptId,
    input_id: InputId,
    *,
    variation_pair_id: VariationPairId | None = None,
    variation_pair: VariationPair | None = None,
) -> ConceptPipelineDataset:
    concept = make_concept(concept_id)
    variations = None
    if variation_pair_id is not None and variation_pair is not None:
        variations = {concept_id: {input_id: {variation_pair_id: variation_pair}}}
    elif variation_pair_id is not None or variation_pair is not None:
        raise ValueError(
            "variation_pair_id and variation_pair must be provided together"
        )
    return ConceptPipelineDataset(
        dataset_name="ds",
        input_template="{vary}",
        input_parameters={"vary": "x"},
        varying_input_param_name="vary",
        varying_inputs={input_id: "text"},
        sanitized_varying_inputs={input_id: "text"},
        concepts=[concept],
        deduplicated_concepts=[concept],
        final_concepts=[concept],
        variations=variations,
    )


def make_stage_baseline(
    concept_id: ConceptId,
    input_id: InputId,
    *,
    baseline_map: dict | None,
    unverbalized=None,
) -> StageResults:
    return StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_id]},
        concepts_at_stage_start=[concept_id],
        concept_verbalization_on_baseline_responses=baseline_map,
        concept_ids_unverbalized_on_baseline=unverbalized,
    )


def make_stage_variation(
    concept_id: ConceptId,
    input_id: InputId,
    *,
    var_map,
    significant_concepts=None,
) -> StageResults:
    return StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [input_id]},
        concepts_at_stage_start=[concept_id],
        variation_bias_results=var_map,
        significant_concepts=significant_concepts or [concept_id],
    )


def make_bias_result(
    input_id: InputId,
    pair_id: VariationPairId,
    responses: VariationPairResponses,
) -> ConceptBiasTestResult:
    # Tests assume the pipeline's current semantics: exactly one response per side per pair.
    assert len(responses.positive_responses) == 1
    assert len(responses.negative_responses) == 1
    assert len(responses.positive_acceptances) == 1
    assert len(responses.negative_acceptances) == 1
    return ConceptBiasTestResult(
        responses_by_input={input_id: {pair_id: responses}},
        flipped_variation_pairs={
            input_id: {pair_id: responses.has_flipped_acceptance()}
        },
    )


# --- Result builders ---
def baseline_flags(verbalized: bool, witness: str = "") -> VerbalizationCheckResult:
    return VerbalizationCheckResult(verbalized=verbalized, witness=witness)


def variation_verbalization(
    pos_map: dict[ResponseId, VerbalizationCheckResult],
    neg_map: dict[ResponseId, VerbalizationCheckResult],
) -> VariationPairVerbalization:
    return VariationPairVerbalization(
        positive_variation_responses_verbalizations=pos_map,
        negative_variation_responses_verbalizations=neg_map,
    )
