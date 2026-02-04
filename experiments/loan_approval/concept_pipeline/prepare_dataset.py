# %%

import asyncio
import concurrent.futures

from dotenv import load_dotenv
from prompts_and_parse_functions import (LOAN_APPROVAL_INPUT_TEMPLATE,
                                         LOAN_VARIATIONS_GENERATION_PROMPT,
                                         LOAN_VARIATIONS_SYSTEM_PROMPT,
                                         parse_loan_variations_response)

from biases_in_the_blind_spot.concept_pipeline.concept_deduplicator import \
    ConceptDeduplicator
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset_preparer import \
    ConceptPipelineDatasetPreparer
from biases_in_the_blind_spot.concept_pipeline.implicit_concept_detector import \
    ImplicitConceptDetector
from biases_in_the_blind_spot.concept_pipeline.input_clusterer import \
    InputClusterer
from biases_in_the_blind_spot.concept_pipeline.variations_generator import \
    VariationsGenerator
from biases_in_the_blind_spot.datasets.loan_approval.loaders import \
    load_loan_approval_dataset
from biases_in_the_blind_spot.util import LOAN_APPROVAL_RESULTS_PATH

load_dotenv()

# %%

dataset_name = "loan_approval"

# %%

# Load all loan applications
loan_applications = load_loan_approval_dataset()
print(f"Loaded {len(loan_applications)} loan applications")

# %%

variations_generator = VariationsGenerator(
    system_prompt=LOAN_VARIATIONS_SYSTEM_PROMPT,
    generation_prompt=LOAN_VARIATIONS_GENERATION_PROMPT,
    result_parser=parse_loan_variations_response,
)
implicit_concept_detector = ImplicitConceptDetector()
input_clusterer = InputClusterer()
concept_deduplicator = ConceptDeduplicator()

input_template = LOAN_APPROVAL_INPUT_TEMPLATE
assert isinstance(input_template, str) and len(input_template) > 0

varying_input_param_name = "loan_application"

# Build varying inputs from enhanced descriptions
varying_inputs = [record.enhanced_description for record in loan_applications]
assert len(varying_inputs) > 0

# No fixed parameters besides the varying loan application
input_parameters: dict[str, str] = {}
# %%

# 2. Instantiate the main pipeline
output_dir = LOAN_APPROVAL_RESULTS_PATH
preparer = ConceptPipelineDatasetPreparer(
    variations_generator=variations_generator,
    implicit_concept_detector=implicit_concept_detector,
    input_clusterer=input_clusterer,
    output_dir=output_dir,
    parsed_labels_mapping={1: "Approved", 0: "Rejected"},
    concept_deduplicator=concept_deduplicator,
)

# %%

try:
    # Try to get existing event loop (e.g. in Jupyter)
    loop = asyncio.get_running_loop()

    def _thread_runner():
        return asyncio.run(
            preparer.prepare_dataset(
                input_template,
                input_parameters,
                varying_input_param_name,
                varying_inputs=varying_inputs,
                dataset_name=dataset_name
            )
        )

    # Run the async function in a thread to avoid "already running loop" errors
    with concurrent.futures.ThreadPoolExecutor() as ex:
        fut = ex.submit(_thread_runner)
        dataset = fut.result()
except RuntimeError as e:
    import traceback
    traceback.print_exc()
    print(f"RuntimeError: {e}")
    if "no running event loop" in str(e):
        # No running event loop – safe to run directly.
        dataset = asyncio.run(
            preparer.prepare_dataset(
                input_template,
                input_parameters,
                varying_input_param_name,
                varying_inputs=varying_inputs,
                dataset_name=dataset_name
            )
        )

# %%

# Print deduplication statistics
print("\n" + "="*80)
print("DEDUPLICATION STATISTICS")
print("="*80)

if dataset.concepts is not None:
    print(f"\nOriginal concepts: {len(dataset.concepts)}")

if dataset.deduplicated_concepts is not None:
    print(f"Deduplicated concepts: {len(dataset.deduplicated_concepts)}")

if dataset.concepts is not None and dataset.deduplicated_concepts is not None:
    removed = len(dataset.concepts) - len(dataset.deduplicated_concepts)
    print(f"Concepts removed: {removed}")

if dataset.duplicate_groups is not None and len(dataset.duplicate_groups) > 0:
    print(f"\nDuplicate groups found: {len(dataset.duplicate_groups)}")

    for i, group in enumerate(dataset.duplicate_groups, 1):
        print("\n" + "━" * 80)
        print(f"DUPLICATE GROUP {i} - {len(group)} CONCEPTS")
        print("━" * 80)

        for concept_id in group:
            # Search in original concepts list since duplicate_groups includes removed concepts
            if dataset.concepts is not None:
                for c in dataset.concepts:
                    if c.id == concept_id:
                        print(f"\n  Concept ID: {concept_id}")
                        print(f"  Title: {c.title}")
                        print(f"  Verbalization Check Guide: {c.verbalization_check_guide}")
                        print(f"  Removal Action: {c.removal_action}")
                        print(f"  Addition Action: {c.addition_action}")
                        print()
                        break

        print()  # Extra spacing after each group
else:
    print("\nNo duplicate groups found")

if dataset.deduplication_comparisons is not None:
    total_comparisons = len(dataset.deduplication_comparisons)
    duplicate_comparisons = sum(1 for c in dataset.deduplication_comparisons if c.duplicate)
    different_comparisons = total_comparisons - duplicate_comparisons

    print(f"\nTotal pairwise comparisons: {total_comparisons}")
    print(f"  - Marked as duplicates: {duplicate_comparisons}")
    print(f"  - Marked as different: {different_comparisons}")
    if total_comparisons > 0:
        dup_pct = (duplicate_comparisons / total_comparisons) * 100
        print(f"  - Duplicate rate: {dup_pct:.1f}%")

print("\n" + "="*80)

# %%
