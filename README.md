# Biases in the Blind Spot: Detecting What LLMs Fail to Mention

Anonymized code for the paper "Biases in the Blind Spot: Detecting What LLMs Fail to Mention".

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies:
   ```bash
   uv sync
   ```

## Code Structure

The code is organized into the following directories:

- `biases_in_the_blind_spot`: The main code for the concept pipeline.
- `experiments`: The experiments for the paper.
- `tests`: The tests for the code.

### Running experiments

This repo already contains our results json for each experiment in the `results` directory.

If you would like to run the experiments yourself, each experiment requires 2 steps to run:

- first, run the `prepare_dataset.py` script to set up initial dataset / generate concepts
- then, run the `concept_pipeline.py` script to run the pipeline
  - NOTE: You need to set the `model_name` in the `concept_pipeline.py` script to the model you want to use

Results are saved in the `results` directory as JSON files. You should also create a `.env` file in your root directory with API keys for Anthropic, OpenAI, and OpenRouter, or you can set environment variables for the same.

### Synthetic datasets

The code to generate our synthetic loan approval and university admission datasets is in the `biases_in_the_blind_spot/datasets` directory. If you would like to extend these datasets, you can rerun the generation scripts with a larger number of generated samples.

**Requirements:** Set your OpenAI API key via `OPENAI_API_KEY` environment variable or in a `.env` file.

**Loan Approval Dataset:**

```bash
# Generate with defaults (10,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset

# Generate a custom number of samples
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset -n 5000

# See all options
uv run python -m biases_in_the_blind_spot.datasets.loan_approval.generate_loan_approval_dataset --help
```

**University Admission Dataset:**

```bash
# Generate with defaults (1,000 samples)
uv run python -m biases_in_the_blind_spot.datasets.university_admission.generate_university_admission_dataset

# Generate a custom number of samples
uv run python -m biases_in_the_blind_spot.datasets.university_admission.generate_university_admission_dataset -n 2000
```

**Common options:** `-n` (number of samples), `-o` (output file), `-m` (model name), `--seed` (random seed).

## Development

### Code Quality

This project uses:

- **Ruff** for linting and code formatting
- **Pyright** for type checking
- **Pytest** for testing

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run pyright .

# Testing
uv run pytest
```
