import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dataclass_wizard import JSONWizard

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept import Concept

# Import at runtime since dataclass_wizard needs to resolve the type during deserialization
from biases_in_the_blind_spot.concept_pipeline.concept_deduplicator import (
    DeduplicationComparison,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_clusters import InputClusters
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variations_quality_checker import (
    ConceptQualitySummary,
    VariationQualityCheck,
)
from biases_in_the_blind_spot.util import ensure_decompressed

if TYPE_CHECKING:
    pass


@dataclass
class ConceptPipelineDataset(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    dataset_name: str

    # Raw inputs
    input_template: str
    input_parameters: dict[str, str]
    varying_input_param_name: str
    varying_inputs: dict[InputId, str]  # input index -> input text

    # Mapping from parsed acceptance values to human-readable labels, e.g., {1: "YES", 0: "NO"}
    parsed_labels_mapping: dict[int, str] | None = None

    # Subclasses configs
    variations_generator_config: dict[str, Any] | None = None
    implicit_concept_detector_config: dict[str, Any] | None = None
    input_sanitizer_config: dict[str, Any] | None = None

    # Concept hypothesis sampling metadata
    concept_hypothesis_sampling_seed: int | None = None
    concept_hypothesis_input_indices: list[InputId] | None = None

    # Sanitized varying inputs
    sanitized_varying_inputs: dict[InputId, str] | None = (
        None  # input index -> sanitized input text
    )
    # Embedding-based clustering of sanitized inputs
    input_clusters: InputClusters | None = None

    # Number of variations per concept per input
    n_variations_per_concept_per_input: int | None = None

    # Removal interpretation mode ("absence" or "negation")
    removal_interpreted_as: Literal["absence", "negation"] | None = None

    # Hypothesized concepts
    concepts: list[Concept] | None = None
    # Deduplicated concepts (after removing duplicates identified by ConceptDeduplicator)
    # If no deduplicator is used, this will be a copy of concepts
    deduplicated_concepts: list[Concept] | None = None
    # Record of all deduplication comparisons made
    deduplication_comparisons: list[DeduplicationComparison] | None = None
    # Groups of duplicate concepts (each group contains concept IDs that are duplicates of each other)
    # Only includes groups with 2+ concepts
    duplicate_groups: list[list[ConceptId]] | None = None
    n_representative_input_clusters: int | None = None
    n_inputs_per_representative_cluster_for_concept_hypothesis: int | None = None

    variations: (
        dict[ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]] | None
    ) = None  # concept_id (only if the concept id was unverbalized in baseline) -> input index -> pair id -> positive and negative variation

    # Quality checks for variations (populated by VariationsQualityChecker)
    variations_quality_checks: (
        dict[ConceptId, dict[InputId, dict[VariationPairId, VariationQualityCheck]]]
        | None
    ) = None

    # Summary of quality for each concept
    variations_quality_summaries: dict[ConceptId, ConceptQualitySummary] | None = None

    # Configuration for quality checker
    variations_quality_checker_config: dict[str, Any] | None = None

    # Final concepts after filtering by variation quality
    # Only includes concepts whose variations meet the quality threshold
    final_concepts: list[Concept] | None = None

    def get_pipeline_concepts(self) -> list[Concept]:
        """Return the concepts to use for the pipeline.

        Returns final_concepts (quality-filtered concepts).

        Raises:
            AssertionError: If final_concepts is not set.
        """
        assert self.final_concepts is not None, "final_concepts is not set"
        return self.final_concepts

    def get_concept_title(self, concept_id: ConceptId) -> str:
        """Return the title for a concept id. Raises if not found."""
        concept_id = ConceptId(concept_id)
        # Use deduplicated concepts if available, otherwise fall back to original concepts
        concepts_to_search = (
            self.deduplicated_concepts
            if self.deduplicated_concepts is not None
            else self.concepts
        )
        assert concepts_to_search is not None and len(concepts_to_search) > 0
        for c in concepts_to_search:
            if c.id == concept_id:
                return c.title
        raise KeyError(f"Concept id not found: {concept_id}")

    def get_input_params(self, input_index: InputId) -> dict[str, str]:
        assert self.sanitized_varying_inputs is not None, (
            "Sanitized varying inputs are not set"
        )
        assert input_index in self.sanitized_varying_inputs, (
            f"Input index has not been sanitized: {input_index}"
        )
        return {
            **self.input_parameters,
            self.varying_input_param_name: self.sanitized_varying_inputs[input_index],
        }

    def select_inputs_from_each_representative_cluster(
        self,
        k: int,
        output_dir: str | Path,
        whitelisted_input_ids: list[InputId] | None = None,
    ) -> dict[ClusterId, list[InputId]]:
        """Return up to k inputs per cluster nearest to that cluster's representative.

        Assumptions:
        - `input_clusters` contains embeddings, clusters, and representatives
        - `k` is a positive integer
        - `whitelisted_input_ids` input ids to filter out from clusters
        """
        assert self.input_clusters is not None
        assert isinstance(k, int) and k >= 1
        reps = self.input_clusters.representatives
        clusters = self.input_clusters.clusters
        embs = self.input_clusters.load_embeddings(output_dir)
        assert isinstance(reps, list) and isinstance(clusters, list)
        assert len(reps) == len(clusters)

        def _l2_sq(a: list[float], b: list[float]) -> float:
            assert isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
            return float(sum((x - y) * (x - y) for x, y in zip(a, b, strict=False)))

        selections: dict[ClusterId, list[InputId]] = {}
        for cluster_idx, members in enumerate(clusters):
            assert isinstance(members, list) and len(members) > 0
            rep_id = reps[cluster_idx]
            rep_vec = embs.get(rep_id)
            assert isinstance(rep_vec, list) and len(rep_vec) > 0
            dist_pairs: list[tuple[InputId, float]] = []
            for member_id in members:
                if (
                    whitelisted_input_ids is not None
                    and member_id not in whitelisted_input_ids
                ):
                    continue
                dist_pairs.append((member_id, _l2_sq(embs[member_id], rep_vec)))
            dist_pairs.sort(key=lambda x: x[1])
            selections[ClusterId(cluster_idx)] = [
                member_id for member_id, _ in dist_pairs[:k]
            ]

        return selections

    def save(self, output_dir: Path):
        dataset_path = ConceptPipelineDataset._get_dataset_path(
            self.dataset_name, output_dir
        )
        data = self.to_json(indent=4, ensure_ascii=False)
        tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dataset_path)

    def get_path(self, output_dir: Path) -> Path:
        return output_dir / f"dataset_{self.dataset_name}.json"

    @staticmethod
    def _get_dataset_path(dataset_name: str, output_dir: Path) -> Path:
        return output_dir / f"dataset_{dataset_name}.json"

    @staticmethod
    def load_by_path(dataset_path: Path) -> "ConceptPipelineDataset | None":
        # Check if the file exists, or if a .gz version exists and decompress it
        try:
            dataset_path = ensure_decompressed(dataset_path)
        except FileNotFoundError:
            return None

        print(f"Loading dataset from {dataset_path.absolute()}")
        with open(dataset_path, encoding="utf-8") as f:
            dataset_data = f.read()
        print("Loaded result data")
        dataset = ConceptPipelineDataset.from_json(dataset_data)
        print("Loaded dataset into object")
        assert isinstance(dataset, ConceptPipelineDataset)

        return dataset

    @staticmethod
    def load_by_name(
        dataset_name: str, output_dir: Path
    ) -> "ConceptPipelineDataset | None":
        dataset_path = ConceptPipelineDataset._get_dataset_path(
            dataset_name, output_dir
        )
        return ConceptPipelineDataset.load_by_path(dataset_path)
