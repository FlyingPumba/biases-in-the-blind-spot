# %%

import asyncio
import concurrent.futures

from dotenv import load_dotenv
from prompts_and_parse_functions import parse_cot_response

from biases_in_the_blind_spot.concept_pipeline.bias_tester import BiasTester
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline import \
    ConceptPipeline
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import \
    ConceptPipelineDataset
from biases_in_the_blind_spot.concept_pipeline.responses_generator import \
    ResponsesGenerator
from biases_in_the_blind_spot.concept_pipeline.verbalization_detector import \
    VerbalizationDetector
from biases_in_the_blind_spot.util import LOAN_APPROVAL_RESULTS_PATH

load_dotenv()

# %%
# model_name = "google/gemma-3-12b-it"
# model_name = "google/gemma-3-27b-it"
# model_name = "google/gemini-2.5-flash"
# model_name = "mistralai/mistral-small-3.1-24b-instruct"
# model_name = "qwen/qwq-32b"
# model_name = "anthropic/claude-sonnet-4-20250514"
model_name = "Qwen/Qwen2.5-32B-Instruct"

dataset_name = "loan_approval"
experiment_key = f"{dataset_name}_{model_name.split('/')[-1]}"
debug_n_concepts = None # 5
debug_n_inputs = None # 200

# %%

# Load dataset
output_dir = LOAN_APPROVAL_RESULTS_PATH
dataset = ConceptPipelineDataset.load_by_name(dataset_name, output_dir)
assert isinstance(dataset, ConceptPipelineDataset)

# %%

# 1. Instantiate pipeline components
responses_generator = ResponsesGenerator(
    model_name=model_name,
    use_anthropic_batches=model_name.startswith("anthropic"),
    use_openai_batches=model_name.startswith("openai"),
)

verbalization_detector = VerbalizationDetector(use_openai_batches=True)

# Bias tester using approval rate as acceptance (1=approve, 0=reject)
varying_input_param_name = "loan_application"
bias_tester = BiasTester(
    responses_generator=responses_generator,
    parse_response_fn=parse_cot_response,
    variating_input_parameter=varying_input_param_name,
)

def representative_inputs_k_per_stage_index_fn(stage_index: int) -> int:
    # We start at 20, then double with each stage.
    if stage_index == 0:
        return 20
    return 2 * representative_inputs_k_per_stage_index_fn(stage_index - 1)

# %%

# 2. Instantiate the main pipeline
pipeline = ConceptPipeline(
    dataset=dataset,
    responses_generator=responses_generator,
    verbalization_detector=verbalization_detector,
    bias_tester=bias_tester,
    representative_inputs_k_per_stage_index_fn=representative_inputs_k_per_stage_index_fn,
    output_dir=output_dir,
    parsed_labels_mapping={1: "Approved", 0: "Rejected"},
)

# %%

# 3. Run the pipeline


try:
    # Try to get existing event loop (e.g. in Jupyter)
    loop = asyncio.get_running_loop()

    def _thread_runner():
        return asyncio.run(
            pipeline.run(
                experiment_key=experiment_key,
                debug_n_concepts=debug_n_concepts,
                debug_n_inputs=debug_n_inputs,
            )
        )

    # Run the async function in a thread to avoid "already running loop" errors
    with concurrent.futures.ThreadPoolExecutor() as ex:
        fut = ex.submit(_thread_runner)
        pipeline_result = fut.result()
except RuntimeError as e:
    import traceback
    traceback.print_exc()
    print(f"RuntimeError: {e}")
    if "no running event loop" in str(e):
        # No running event loop – safe to run directly.
        pipeline_result = asyncio.run(
            pipeline.run(
                experiment_key=experiment_key,
                debug_n_concepts=debug_n_concepts,
                debug_n_inputs=debug_n_inputs,
            )
        )

# %%
