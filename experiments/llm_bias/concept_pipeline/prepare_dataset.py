# %%

import asyncio
import concurrent.futures

from dotenv import load_dotenv

from experiments.llm_bias.concept_pipeline.prompts_and_parse_functions import (
    load_resumes_with_pronouns,
)
from biases_in_the_blind_spot.concept_pipeline.concept_deduplicator import (
    ConceptDeduplicator,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset_preparer import (
    ConceptPipelineDatasetPreparer,
)
from biases_in_the_blind_spot.concept_pipeline.implicit_concept_detector import (
    ImplicitConceptDetector,
)
from biases_in_the_blind_spot.concept_pipeline.input_clusterer import InputClusterer
from biases_in_the_blind_spot.concept_pipeline.input_sanitizer import InputSanitizer
from biases_in_the_blind_spot.concept_pipeline.variations_generator import (
    VariationsGenerator,
)
from biases_in_the_blind_spot.util import (
    LLM_BIAS_PROMPTS_PATH,
    CONCEPT_PIPELINE_LLM_BIAS_RESULTS_PATH,
)

load_dotenv()

# %%

dataset_name = "llm_bias"

# %%

# resumes = load_raw_resumes()
resumes = load_resumes_with_pronouns(downsample=False)
print(f"Loaded {len(resumes)} resumes")

# job_description_name = "base_description"
job_description_name = "meta_job_description"

with open(
    LLM_BIAS_PROMPTS_PATH / "job_descriptions" / f"{job_description_name}.txt",
    encoding="utf-8",
) as f:
    job_description = f.read()

# %%

variations_generator = VariationsGenerator(use_openai_batches=True)
implicit_concept_detector = ImplicitConceptDetector()
input_sanitizer = InputSanitizer(use_openai_batches=True)
input_clusterer = InputClusterer()
concept_deduplicator = ConceptDeduplicator(use_openai_batches=True)

# %%

input_template_path = LLM_BIAS_PROMPTS_PATH / "system_prompts" / "yes_no_long_cot.txt"
with open(input_template_path, "r", encoding="utf-8") as f:
    input_template = f.read()
assert isinstance(input_template, str) and len(input_template) > 0

input_parameters = {
    "anti_bias_statement": "",
    "job_description": job_description,
}
varying_input_param_name = "resume"

# %%

# 2. Instantiate the main pipeline
output_dir = CONCEPT_PIPELINE_LLM_BIAS_RESULTS_PATH
preparer = ConceptPipelineDatasetPreparer(
    variations_generator=variations_generator,
    implicit_concept_detector=implicit_concept_detector,
    input_clusterer=input_clusterer,
    output_dir=output_dir,
    parsed_labels_mapping={1: "YES", 0: "NO"},
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
                varying_inputs=resumes,
                dataset_name=dataset_name,
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
                varying_inputs=resumes,
                dataset_name=dataset_name,
            )
        )
