import gzip
from pathlib import Path
from typing import TypeVar

import torch

T = TypeVar("T")
K = TypeVar("K")


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT_PATH = Path(__file__).parent.parent

EXPERIMENTS_PATH = PROJECT_ROOT_PATH / "experiments"
CONCEPT_PIPELINE_LLM_BIAS_RESULTS_PATH = (
    EXPERIMENTS_PATH / "llm_bias" / "concept_pipeline" / "results"
)
CONCEPT_PIPELINE_SPORTS_UNDERSTANDING_RESULTS_PATH = (
    EXPERIMENTS_PATH / "sports_understanding" / "concept_pipeline" / "results"
)
LOAN_APPROVAL_RESULTS_PATH = (
    EXPERIMENTS_PATH / "loan_approval" / "concept_pipeline" / "results"
)
UNIVERSITY_ADMISSION_RESULTS_PATH = (
    EXPERIMENTS_PATH / "university_admission" / "concept_pipeline" / "results"
)
AITA_RESULTS_PATH = EXPERIMENTS_PATH / "aita" / "concept_pipeline" / "results"

LLM_BIAS_ROOT_PATH = Path(__file__).parent / "llm_bias"
LLM_BIAS_DATA_PATH = LLM_BIAS_ROOT_PATH / "data"
LLM_BIAS_PROMPTS_PATH = LLM_BIAS_ROOT_PATH / "prompts"


def cos_sims(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity between two tensors."""
    a_normed = a / a.norm(dim=-1, keepdim=True)
    b_normed = b / b.norm(dim=-1, keepdim=True)
    return torch.matmul(a_normed, b_normed.T)


def ensure_decompressed(file_path: Path) -> Path:
    """Ensure a file is decompressed from .gz if needed.

    If the file doesn't exist but a .gz version does, decompresses it.
    Returns the path to the (possibly newly decompressed) file.

    Args:
        file_path: Path to the file (without .gz extension)

    Returns:
        The file_path if it exists or was successfully decompressed

    Raises:
        FileNotFoundError: If neither the file nor its .gz version exists
    """
    if file_path.exists():
        return file_path

    gz_path = Path(str(file_path) + ".gz")
    if gz_path.exists():
        print(f"Found compressed file at {gz_path}, decompressing...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(file_path, "wb") as f_out:
                f_out.write(f_in.read())
        print(f"Decompressed to {file_path}")
        return file_path

    raise FileNotFoundError(f"Neither {file_path} nor {gz_path} exists")
