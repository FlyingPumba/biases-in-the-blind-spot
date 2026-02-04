# isort: skip_file

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.unfaithful_concepts import (
    get_unfaithful_concepts,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)

from tests.concept_pipeline.helpers import baseline_flags, make_dataset


def test_unfaithful_concepts_threshold_is_ge_and_or_semantics():
    """A concept is NOT unfaithful if flipped-pair verbalization ratio == threshold.

    A flipped pair counts as verbalized if either positive OR negative response is verbalized.
    """
    cid = ConceptId()
    iid = InputId()
    pid1 = VariationPairId()
    pid2 = VariationPairId()
    rid_p1 = ResponseId()
    rid_n1 = ResponseId()
    rid_p2 = ResponseId()
    rid_n2 = ResponseId()

    # Two flipped pairs (acceptances differ).
    pair1 = VariationPairResponses(
        positive_responses={rid_p1: "p1"},
        negative_responses={rid_n1: "n1"},
        positive_acceptances={rid_p1: 1},
        negative_acceptances={rid_n1: 0},
    )
    pair2 = VariationPairResponses(
        positive_responses={rid_p2: "p2"},
        negative_responses={rid_n2: "n2"},
        positive_acceptances={rid_p2: 1},
        negative_acceptances={rid_n2: 0},
    )

    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        significant_concepts=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        concept_verbalization_on_baseline_responses={
            cid: {iid: {ResponseId(): baseline_flags(False)}}
        },
        variation_bias_results={
            cid: ConceptBiasTestResult(
                responses_by_input={iid: {pid1: pair1, pid2: pair2}},
                flipped_variation_pairs={iid: {pid1: True, pid2: True}},
            )
        },
        concept_verbalization_on_variation_responses={
            cid: {
                iid: {
                    pid1: VariationPairVerbalization(
                        # OR semantics: negative verbalized => pair counts as verbalized
                        positive_variation_responses_verbalizations={
                            rid_p1: baseline_flags(False)
                        },
                        negative_variation_responses_verbalizations={
                            rid_n1: baseline_flags(True, "w")
                        },
                    ),
                    pid2: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_p2: baseline_flags(False)
                        },
                        negative_variation_responses_verbalizations={
                            rid_n2: baseline_flags(False)
                        },
                    ),
                }
            }
        },
    )

    dataset = make_dataset(cid, iid)
    result = ConceptPipelineResult(
        stages=[stage],
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
    )

    # 1/2 flipped pairs verbalized => ratio == 0.5 == threshold => NOT unfaithful.
    assert get_unfaithful_concepts(dataset, result, stage) == []


def test_unfaithful_concepts_below_threshold_is_unfaithful():
    cid = ConceptId()
    iid = InputId()
    pid1 = VariationPairId()
    pid2 = VariationPairId()
    rid_p1 = ResponseId()
    rid_n1 = ResponseId()
    rid_p2 = ResponseId()
    rid_n2 = ResponseId()

    pair1 = VariationPairResponses(
        positive_responses={rid_p1: "p1"},
        negative_responses={rid_n1: "n1"},
        positive_acceptances={rid_p1: 1},
        negative_acceptances={rid_n1: 0},
    )
    pair2 = VariationPairResponses(
        positive_responses={rid_p2: "p2"},
        negative_responses={rid_n2: "n2"},
        positive_acceptances={rid_p2: 1},
        negative_acceptances={rid_n2: 0},
    )

    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        significant_concepts=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        concept_verbalization_on_baseline_responses={
            cid: {iid: {ResponseId(): baseline_flags(False)}}
        },
        variation_bias_results={
            cid: ConceptBiasTestResult(
                responses_by_input={iid: {pid1: pair1, pid2: pair2}},
                flipped_variation_pairs={iid: {pid1: True, pid2: True}},
            )
        },
        concept_verbalization_on_variation_responses={
            cid: {
                iid: {
                    pid1: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_p1: baseline_flags(False)
                        },
                        negative_variation_responses_verbalizations={
                            rid_n1: baseline_flags(True, "w")
                        },
                    ),
                    pid2: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={
                            rid_p2: baseline_flags(False)
                        },
                        negative_variation_responses_verbalizations={
                            rid_n2: baseline_flags(False)
                        },
                    ),
                }
            }
        },
    )

    dataset = make_dataset(cid, iid)
    result = ConceptPipelineResult(
        stages=[stage],
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.6,
    )

    # 1/2 = 0.5 < 0.6 => unfaithful.
    assert get_unfaithful_concepts(dataset, result, stage) == [cid]
