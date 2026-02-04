import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dataclass_wizard import JSONWizard

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
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


@dataclass
class StageResults(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    # Stage metadata
    stage_idx: int
    k_inputs_per_representative_cluster: int
    seed: int  # selection reproducibility
    # Per representative cluster → list of absolute input indices
    input_indices_by_representative_cluster: dict[ClusterId, list[InputId]]

    concepts_at_stage_start: list[ConceptId]
    # Target alpha for significant concepts in this stage
    stage_significant_concepts_p_value_alpha: float | None = None
    # O'Brien-Fleming alpha threshold used to check for early stopping in this stage
    early_stop_alpha: float | None = None

    concept_verbalization_on_baseline_responses: (
        dict[ConceptId, dict[InputId, dict[ResponseId, VerbalizationCheckResult]]]
        | None
    ) = None  # concept_id -> input index -> response index -> verbalization result (flag + witness)
    concept_ids_unverbalized_on_baseline: list[ConceptId] | None = None

    variation_bias_results: dict[ConceptId, ConceptBiasTestResult] | None = None
    significant_concepts: list[ConceptId] | None = None
    early_stopped_concepts: list[ConceptId] | None = (
        None  # Concepts stopped for efficacy (O'Brien-Fleming)
    )
    futility_stopped_concepts: list[ConceptId] | None = (
        None  # Concepts stopped for futility (conditional power)
    )
    # Conditional power estimates computed during futility stopping (concept_id -> power).
    # Only populated for concepts where conditional power is actually computed.
    conditional_power_by_concept: dict[ConceptId, float] | None = None

    concept_verbalization_on_variation_responses: (
        dict[
            ConceptId,
            dict[InputId, dict[VariationPairId, VariationPairVerbalization]],
        ]
        | None
    ) = None  # concept_id -> input_index -> pair_index -> verbalization for the responses of the positive and negative variations in the pair
    concepts_at_stage_end: list[ConceptId] | None = None

    def get_stage_input_ids(self) -> list[InputId]:
        input_ids_in_stage = [
            input_idx
            for input_indices_in_cluster in self.input_indices_by_representative_cluster.values()
            for input_idx in input_indices_in_cluster
        ]
        # Validate that each cluster has at most k_inputs_per_representative_cluster
        for (
            cluster_id,
            input_indices,
        ) in self.input_indices_by_representative_cluster.items():
            assert len(input_indices) <= self.k_inputs_per_representative_cluster, (
                f"Cluster {cluster_id} has {len(input_indices)} inputs, expected at most {self.k_inputs_per_representative_cluster}"
            )
        return input_ids_in_stage

    def get_concept_ids_at_stage_start(self) -> list[ConceptId]:
        return list(
            {ConceptId(concept_id) for concept_id in self.concepts_at_stage_start}
        )

    def __post_init__(self) -> None:
        """Validate that input-indexed fields only reference this stage's inputs.

        This runs after (de)serialization and does not mutate stored data; it
        asserts that stage-scoped structures do not contain stale input indices.
        """
        stage_inputs = set(self.get_stage_input_ids())
        assert len(stage_inputs) > 0

        print(
            f"Stage {self.stage_idx}: validating input-indexed fields (no mutation, {len(stage_inputs)} inputs)"
        )

        # 1) Baseline verbalization flags: concept_id -> input_idx -> resp_idx -> bool
        if self.concept_verbalization_on_baseline_responses is not None:
            for (
                concept_id,
                by_input,
            ) in self.concept_verbalization_on_baseline_responses.items():
                stale_inputs = [i for i in by_input.keys() if i not in stage_inputs]
                assert len(stale_inputs) == 0, (
                    f"Stage {self.stage_idx}: baseline verbalization has stale inputs for concept {concept_id}: {stale_inputs}"
                )

        # 2) Variation verbalization flags: ConceptSide -> input_idx -> pair_idx -> ...
        if self.concept_verbalization_on_variation_responses is not None:
            for (
                concept_id,
                by_input,
            ) in self.concept_verbalization_on_variation_responses.items():
                stale_inputs = [i for i in by_input.keys() if i not in stage_inputs]
                assert len(stale_inputs) == 0, (
                    f"Stage {self.stage_idx}: variation verbalization has stale inputs for concept {concept_id}: {stale_inputs}"
                )

        # 3) Variation bias results: per concept id -> ConceptBiasTestResult
        if self.variation_bias_results is not None:
            for concept_id, res in self.variation_bias_results.items():
                stale_inputs = [
                    i
                    for i in list(res.responses_by_input.keys())
                    if i not in stage_inputs
                ]
                assert len(stale_inputs) == 0, (
                    f"Stage {self.stage_idx}: variation bias results have stale inputs for concept {concept_id}: {stale_inputs}"
                )

                for input_idx, responses_by_pair in res.responses_by_input.items():
                    assert len(responses_by_pair) > 0, (
                        f"Stage {self.stage_idx}: empty variation pairs for concept {concept_id} input {input_idx}"
                    )
                    for pair_idx, pair_responses in responses_by_pair.items():
                        assert isinstance(pair_responses, VariationPairResponses)
                        assert hasattr(
                            pair_responses, "positive_responses"
                        ) and hasattr(pair_responses, "negative_responses")
                        assert hasattr(
                            pair_responses, "positive_acceptances"
                        ) and hasattr(pair_responses, "negative_acceptances")
                        assert len(pair_responses.positive_responses) > 0, (
                            f"Stage {self.stage_idx}: no positive responses for concept {concept_id} input {input_idx} pair {pair_idx}"
                        )
                        assert len(pair_responses.negative_responses) > 0, (
                            f"Stage {self.stage_idx}: no negative responses for concept {concept_id} input {input_idx} pair {pair_idx}"
                        )

                        for resp_id, resp in pair_responses.positive_responses.items():
                            assert isinstance(resp, str) and len(resp.strip()) > 0, (
                                f"Stage {self.stage_idx}: empty positive response for concept {concept_id} "
                                f"input {input_idx} pair {pair_idx} response {resp_id}"
                            )
                        for resp_id, resp in pair_responses.negative_responses.items():
                            assert isinstance(resp, str) and len(resp.strip()) > 0, (
                                f"Stage {self.stage_idx}: empty negative response for concept {concept_id} "
                                f"input {input_idx} pair {pair_idx} response {resp_id}"
                            )

                flipped = getattr(res, "flipped_variation_pairs", None)
                if flipped is not None:
                    assert set(flipped.keys()) == set(res.responses_by_input.keys()), (
                        f"Stage {self.stage_idx}: flipped_variation_pairs input keys mismatch for concept {concept_id}"
                    )
                    for input_idx, responses_by_pair in res.responses_by_input.items():
                        per_input = flipped[input_idx]
                        assert set(per_input.keys()) == set(responses_by_pair.keys()), (
                            f"Stage {self.stage_idx}: flipped_variation_pairs pair keys mismatch for concept {concept_id} "
                            f"input {input_idx}"
                        )
                        for pair_idx, is_flipped in per_input.items():
                            assert isinstance(is_flipped, bool), (
                                f"Stage {self.stage_idx}: flipped_variation_pairs value is not bool for concept {concept_id} "
                                f"input {input_idx} pair {pair_idx}"
                            )


@dataclass
class ConceptPipelineResult(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    experiment_key: str = "test"

    # Mapping from parsed acceptance values to human-readable labels, e.g., {1: "YES", 0: "NO"}
    parsed_labels_mapping: dict[int, str] | None = None

    # Subclasses configs
    responses_generator_config: dict[str, Any] | None = None
    verbalization_detector_config: dict[str, Any] | None = None
    bias_tester_config: dict[str, Any] | None = None

    # Verbalization and selection thresholds
    baseline_verbalization_threshold: float | None = None
    variations_verbalization_threshold: float | None = None
    # Significance test used to select concepts ("fisher" or "mcnemar")
    significance_test: Literal["fisher", "mcnemar"] | None = None

    # Number of baseline responses per input before pre-filtering
    n_baseline_responses_pre_filter_per_input: int | None = None
    # Number of baseline responses per input
    n_baseline_responses_per_input: int | None = None

    # Baseline responses and acceptances
    baseline_responses_by_input: dict[InputId, dict[ResponseId, str]] | None = (
        None  # input index -> response index -> a baseline response
    )
    pre_filtered_baseline_responses_by_input: (
        dict[InputId, dict[ResponseId, str]] | None
    ) = None  # input index -> response index -> a baseline response
    baseline_acceptances_by_input: (
        dict[InputId, dict[ResponseId, int | None]] | None
    ) = None  # input index -> response index -> a parsed acceptance
    pre_filtered_baseline_acceptances_by_input: (
        dict[InputId, dict[ResponseId, int | None]] | None
    ) = None  # input index -> response index -> a baseline response

    # Filtered varying inputs
    filtered_varying_inputs: list[InputId] | None = None  # list of input indices

    # Early stopping configuration
    futility_stop_power_threshold: float | None = None  # Conditional power threshold

    # Target alpha for significant concepts in the entire pipeline
    significant_concepts_p_value_alpha: float | None = None

    apply_bonferroni_correction: bool | None = (
        None  # Wether to divide the significance threshold by the number of concepts we have tested on
    )

    stages: list[StageResults] | None = None

    # Final selection after the last stage: unfaithful concepts that also pass the significance test
    significant_unfaithful_concepts: list[ConceptId] | None = None

    def get_figures_root(self, output_dir: Path) -> Path:
        figures_root = output_dir / "figures" / self.experiment_key
        os.makedirs(figures_root, exist_ok=True)
        return figures_root

    def get_stage_figures_root(self, output_dir: Path, stage: StageResults) -> Path:
        """Return stage-specific figures directory.

        Uses the stage index to create a nested folder.
        """
        base = self.get_figures_root(output_dir)
        stage_root = base / f"stage-{stage.stage_idx}"
        os.makedirs(stage_root, exist_ok=True)
        return stage_root

    def get_baseline_acceptances_list(self) -> list[int | None]:
        """Flatten baseline acceptances ordered by input index then response index.

        Returns a list of acceptances ordered by input index (ascending) and,
        within each input, by response index (ascending).
        """
        assert self.baseline_acceptances_by_input is not None
        flat: list[int | None] = []
        for input_idx in sorted(self.baseline_acceptances_by_input.keys()):
            per_resp = self.baseline_acceptances_by_input[input_idx]
            assert isinstance(per_resp, dict) and len(per_resp) > 0
            for response_idx in per_resp.keys():
                flat.append(per_resp[response_idx])
        assert len(flat) > 0
        return flat
