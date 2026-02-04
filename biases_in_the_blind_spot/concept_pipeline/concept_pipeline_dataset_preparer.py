import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from biases_in_the_blind_spot.concept_pipeline.concept_deduplicator import (
    ConceptDeduplicator,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.implicit_concept_detector import (
    ImplicitConceptDetector,
)
from biases_in_the_blind_spot.concept_pipeline.input_clusterer import InputClusterer
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.input_sanitizer import InputSanitizer
from biases_in_the_blind_spot.concept_pipeline.variations_generator import (
    VariationsGenerator,
)
from biases_in_the_blind_spot.concept_pipeline.variations_quality_checker import (
    VariationsQualityChecker,
)


@dataclass
class ConceptPipelineDatasetPreparer:
    variations_generator: VariationsGenerator
    implicit_concept_detector: ImplicitConceptDetector
    input_clusterer: InputClusterer
    input_sanitizer: InputSanitizer | None = None
    concept_deduplicator: ConceptDeduplicator | None = None
    variations_quality_checker: VariationsQualityChecker | None = None
    output_dir: Path = Path("results")

    n_variations_per_concept_per_input: int = 1
    n_representative_input_clusters: int = 10
    n_inputs_per_representative_cluster_for_concept_hypothesis: int = 3

    removal_interpreted_as: Literal["absence", "negation"] = "negation"

    # Human-readable labels for parsed acceptance values, e.g., {1: "YES", 0: "NO"}
    parsed_labels_mapping: dict[int, str] | None = None

    def _populate_configs_if_needed(self, dataset: ConceptPipelineDataset):
        if dataset.variations_generator_config is None:
            dataset.variations_generator_config = self.variations_generator.config
        if dataset.implicit_concept_detector_config is None:
            dataset.implicit_concept_detector_config = (
                self.implicit_concept_detector.config
            )
        if dataset.input_sanitizer_config is None and self.input_sanitizer is not None:
            dataset.input_sanitizer_config = self.input_sanitizer.config
        if (
            dataset.variations_quality_checker_config is None
            and self.variations_quality_checker is not None
        ):
            dataset.variations_quality_checker_config = (
                self.variations_quality_checker.config
            )
        if dataset.n_variations_per_concept_per_input is None:
            dataset.n_variations_per_concept_per_input = (
                self.n_variations_per_concept_per_input
            )
        if dataset.n_representative_input_clusters is None:
            dataset.n_representative_input_clusters = (
                self.n_representative_input_clusters
            )
        if dataset.n_inputs_per_representative_cluster_for_concept_hypothesis is None:
            dataset.n_inputs_per_representative_cluster_for_concept_hypothesis = (
                self.n_inputs_per_representative_cluster_for_concept_hypothesis
            )
        if dataset.parsed_labels_mapping is None:
            dataset.parsed_labels_mapping = self.parsed_labels_mapping
        if dataset.removal_interpreted_as is None:
            dataset.removal_interpreted_as = self.removal_interpreted_as

    def _ensure_varying_inputs(
        self, dataset: ConceptPipelineDataset, inputs: list[str]
    ) -> None:
        assert isinstance(inputs, list) and len(inputs) > 0
        assert all(isinstance(r, str) and len(r) > 0 for r in inputs)
        stored_inputs = list(dataset.varying_inputs.values())
        assert len(stored_inputs) == len(inputs)
        for stored, expected in zip(stored_inputs, inputs, strict=True):
            assert stored == expected, "Varying input mismatch"
        dataset.save(self.output_dir)

    async def _sanitize_all_varying_inputs_if_needed(
        self,
        dataset: ConceptPipelineDataset,
    ) -> None:
        assert dataset.varying_inputs is not None
        if dataset.sanitized_varying_inputs is None:
            dataset.sanitized_varying_inputs = {}
        # Build batch of pending inputs that still need sanitization
        pending: dict[InputId, str] = {}
        for idx, original in dataset.varying_inputs.items():
            if idx not in dataset.sanitized_varying_inputs:
                assert isinstance(original, str) and len(original) > 0
                pending[idx] = original

        if len(pending) > 0:
            print(f"Sanitizing {len(pending)} varying inputs")
            if self.input_sanitizer is None:
                cleaned_map = pending
            else:
                cleaned_map = await self.input_sanitizer.sanitize_text_batch(pending)
            # Merge results back
            for idx, cleaned in cleaned_map.items():
                dataset.sanitized_varying_inputs[idx] = cleaned
            # Persist once after batch
            dataset.save(self.output_dir)
        assert len(dataset.sanitized_varying_inputs) == len(dataset.varying_inputs), (
            f"Sanitized varying inputs mismatch: {len(dataset.sanitized_varying_inputs)} != {len(dataset.varying_inputs)}"
        )
        print("Sanitized varying inputs ready")

    async def _hypothesize_concepts_if_needed(
        self,
        dataset: ConceptPipelineDataset,
    ):
        if dataset.concepts is None:
            assert dataset.removal_interpreted_as is not None
            await self.implicit_concept_detector.hypothesize_concepts(
                dataset, self.output_dir, dataset.removal_interpreted_as
            )
            dataset.save(self.output_dir)

        assert dataset.concepts is not None and len(dataset.concepts) > 0
        print(f"Hypothesized {len(dataset.concepts)} concepts:")
        for concept in dataset.concepts:
            print(f" - `{concept.title}`:")
            print(
                f"   - Verbalization check guide: {concept.verbalization_check_guide}"
            )
            print(f"   - Removal action: {concept.removal_action}")
            print(f"   - Addition action: {concept.addition_action}")
        print()

    async def _deduplicate_concepts_if_needed(
        self,
        dataset: ConceptPipelineDataset,
    ):
        """Deduplicate concepts if not already done.

        If concept_deduplicator is provided, use it to identify and remove duplicates.
        Otherwise, deduplicated_concepts is just a copy of concepts.
        """
        if dataset.deduplicated_concepts is not None:
            print(
                f"Concepts already deduplicated: {len(dataset.deduplicated_concepts)} concepts"
            )
            return

        assert dataset.concepts is not None and len(dataset.concepts) > 0

        if self.concept_deduplicator is None:
            # No deduplicator provided, just copy the concepts
            print("No concept deduplicator provided, using all concepts")
            dataset.deduplicated_concepts = list(dataset.concepts)
            dataset.deduplication_comparisons = []
            dataset.duplicate_groups = []
        else:
            # Use the deduplicator to remove duplicates
            print(f"Deduplicating {len(dataset.concepts)} concepts...")
            (
                dataset.deduplicated_concepts,
                dataset.deduplication_comparisons,
                dataset.duplicate_groups,
            ) = await self.concept_deduplicator.deduplicate_concepts(dataset.concepts)

        # Save after deduplication
        dataset.save(self.output_dir)

        print(
            f"Final concept count after deduplication: {len(dataset.deduplicated_concepts)}"
        )

    async def _generate_variations(self, dataset: ConceptPipelineDataset):
        assert dataset.sanitized_varying_inputs is not None
        assert dataset.deduplicated_concepts is not None

        # Build sets of input indices
        input_ids = set(dataset.sanitized_varying_inputs.keys())
        concept_ids = [c.id for c in dataset.deduplicated_concepts]

        # Initialize variations if needed
        if dataset.variations is None:
            dataset.variations = {}

        # Build requested (concept_id, input_idx) pairs, only for missing variations
        pairs_to_generate: list[tuple[ConceptId, InputId]] = []
        for concept_id in concept_ids:
            for input_idx in input_ids:
                # Check how many variations already exist for this pair
                existing_count = 0
                if (
                    concept_id in dataset.variations
                    and input_idx in dataset.variations[concept_id]
                ):
                    existing_count = len(dataset.variations[concept_id][input_idx])

                # Only generate missing variations
                assert dataset.n_variations_per_concept_per_input is not None
                missing_count = (
                    dataset.n_variations_per_concept_per_input - existing_count
                )
                for _ in range(missing_count):
                    pairs_to_generate.append((concept_id, input_idx))

        if len(pairs_to_generate) == 0:
            print("All variations already exist, skipping generation")
            return

        # Convert to set for efficient lookup
        pairs_set = set(pairs_to_generate)

        print(
            f"Generating {len(pairs_to_generate)} missing variations across {len(concept_ids)} concepts"
        )

        # Do batching here only if not using OpenAI batches.
        # If using OpenAI batches, we don't need to batch because @api_llm_base._generate_batch_llm_response_openai does the batching, and sends all requests at once.
        max_batch = (
            2000 if not self.variations_generator.use_openai_batches else sys.maxsize
        )

        total = len(pairs_to_generate)
        start = 0
        while start < total:
            end = min(start + max_batch, total)
            batch_pairs = pairs_to_generate[start:end]
            # Build per-batch map ConceptId -> [input_idx]
            batch_map: dict[ConceptId, list[InputId]] = {}
            for concept_id, idx in batch_pairs:
                batch_map.setdefault(concept_id, []).append(idx)

            print(
                f" - Batch {start // max_batch + 1}/{total // max_batch + 1}: {len(batch_pairs)} pairs"
            )
            assert dataset.removal_interpreted_as is not None
            generated_variations = (
                await self.variations_generator.generate_concept_variations(
                    dataset, batch_map, dataset.removal_interpreted_as
                )
            )

            # Merge generated variations into current stage, but only for requested sides/inputs
            if dataset.variations is None:
                dataset.variations = {}
            for (
                concept_id,
                variations_by_input_for_concept,
            ) in generated_variations.items():
                for (
                    input_idx,
                    variations_by_pair_for_input,
                ) in variations_by_input_for_concept.items():
                    if (concept_id, input_idx) not in pairs_set:
                        continue

                    if concept_id not in dataset.variations:
                        dataset.variations[concept_id] = {}
                    if input_idx not in dataset.variations[concept_id]:
                        dataset.variations[concept_id][input_idx] = {}
                    for pair_id, variation_pair in variations_by_pair_for_input.items():
                        dataset.variations[concept_id][input_idx][pair_id] = (
                            variation_pair
                        )

            # Persist after each batch (partial results)
            dataset.save(self.output_dir)
            start = end

        assert dataset.variations is not None
        print(f"Variations ready for {len(dataset.variations)} concepts")

    async def _check_variations_quality_if_needed(
        self,
        dataset: ConceptPipelineDataset,
    ):
        """Check quality of variations and filter concepts if quality checker is provided."""
        if self.variations_quality_checker is None:
            # No quality checker provided, just copy deduplicated_concepts to final_concepts
            print("No variations quality checker provided, using all concepts")
            dataset.final_concepts = list(dataset.deduplicated_concepts or [])
            return

        if dataset.variations_quality_checks is not None:
            print("Variations quality already checked")
            # Ensure final_concepts is populated
            if (
                dataset.final_concepts is None
                and dataset.deduplicated_concepts is not None
            ):
                assert dataset.variations_quality_summaries is not None
                dataset.final_concepts = (
                    self.variations_quality_checker.filter_concepts_by_quality(
                        dataset.deduplicated_concepts,
                        dataset.variations_quality_summaries,
                    )
                )
            return

        assert dataset.variations is not None
        assert dataset.deduplicated_concepts is not None

        print(
            f"Checking quality of variations for {len(dataset.deduplicated_concepts)} concepts..."
        )

        # Check quality of all variation pairs
        (
            quality_checks,
            quality_summaries,
        ) = await self.variations_quality_checker.check_variations_quality(
            concepts=dataset.deduplicated_concepts,
            variations=dataset.variations,
        )

        # Store results
        dataset.variations_quality_checks = quality_checks
        dataset.variations_quality_summaries = quality_summaries

        # Filter concepts by quality
        dataset.final_concepts = (
            self.variations_quality_checker.filter_concepts_by_quality(
                dataset.deduplicated_concepts,
                quality_summaries,
            )
        )

        # Save after quality check
        dataset.save(self.output_dir)

        print(
            f"Final concept count after quality filtering: {len(dataset.final_concepts)}"
        )

    async def prepare_dataset(
        self,
        input_template: str,
        input_parameters: dict[str, str],
        varying_input_param_name: str,
        varying_inputs: list[str],
        dataset_name: str,
    ) -> ConceptPipelineDataset:
        assert len(varying_inputs) > 0, "No varying inputs provided"
        assert varying_input_param_name not in input_parameters, (
            "Varying input parameter name must not be in input parameters"
        )
        dataset = ConceptPipelineDataset.load_by_name(dataset_name, self.output_dir)
        if dataset is None:
            print(
                f"No dataset found at for {dataset_name} in {self.output_dir}, creating new dataset"
            )
            dataset = ConceptPipelineDataset(
                input_template=input_template,
                input_parameters=input_parameters,
                varying_input_param_name=varying_input_param_name,
                varying_inputs={InputId(): text for text in varying_inputs},
                dataset_name=dataset_name,
            )

        self._populate_configs_if_needed(dataset)
        dataset.save(self.output_dir)

        assert len(varying_inputs) > 0, "No varying inputs provided"
        self._ensure_varying_inputs(dataset, varying_inputs)

        await self._sanitize_all_varying_inputs_if_needed(dataset)

        # Cluster sanitized inputs and select representative indices for concept hypothesis
        await self.input_clusterer.cluster_inputs_if_needed(dataset, self.output_dir)
        assert dataset.input_clusters is not None, "Input clusters are not computed"

        dataset.save(self.output_dir)
        await self._hypothesize_concepts_if_needed(
            dataset,
        )

        # Deduplicate concepts after hypothesis
        await self._deduplicate_concepts_if_needed(dataset)
        dataset.save(self.output_dir)

        await self._generate_variations(
            dataset,
        )
        dataset.save(self.output_dir)

        # Check quality of variations and filter concepts
        await self._check_variations_quality_if_needed(dataset)
        dataset.save(self.output_dir)

        return dataset
