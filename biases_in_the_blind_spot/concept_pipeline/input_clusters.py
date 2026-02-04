import pickle
from dataclasses import dataclass
from pathlib import Path

from dataclass_wizard import JSONWizard

from biases_in_the_blind_spot.concept_pipeline.input_id import InputId


@dataclass
class InputClusters(JSONWizard):
    """Container for embeddings-based input clustering results.

    Assumptions:
    - `embeddings_by_input_index_path` is a relative path (to the pipeline's output_dir)
      to a pickle file containing a dict[InputId, list[float]] of embeddings for every
      sanitized input index used for clustering
    - `clusters` contains exactly one list per cluster, with input indices as found in the result's sanitized inputs
    - `representatives` has the same length as `clusters` and contains one input index per cluster
    """

    embeddings_by_input_index_path: str
    clusters: list[list[InputId]]
    representatives: list[InputId]

    # --- Embeddings IO helpers ---
    def get_embeddings_abs_path(self, output_dir: str | Path) -> Path:
        """Resolve absolute path to the embeddings pickle, under the results root.

        Assumptions:
        - `output_dir` points to the root directory where results for this run are stored
        - `embeddings_by_input_index_path` is a POSIX-like relative path or filename
        """
        base = Path(str(output_dir))
        rel = Path(self.embeddings_by_input_index_path)
        assert not rel.is_absolute()
        return base / rel

    @staticmethod
    def save_embeddings_abs(
        abs_path: str | Path, embeddings_by_input_index: dict[InputId, list[float]]
    ) -> None:
        """Write embeddings dict to a pickle at the given absolute path.

        Assumptions:
        - `embeddings_by_input_index` maps every input index used for clustering to a non-empty list[float]
        - Parent directory for `abs_path` exists or can be created
        """
        p = Path(str(abs_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        assert (
            isinstance(embeddings_by_input_index, dict)
            and len(embeddings_by_input_index) > 0
        )
        with open(p, "wb") as f:
            pickle.dump(embeddings_by_input_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_embeddings_abs(abs_path: str | Path) -> dict[InputId, list[float]]:
        """Load embeddings dict from a pickle at the given absolute path."""
        p = Path(str(abs_path))
        assert p.exists() and p.is_file()
        with open(p, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, dict) and len(data) > 0
        # Lightweight structural check
        k0 = next(iter(data))
        assert isinstance(k0, str)
        v0 = data[k0]
        assert (
            isinstance(v0, list)
            and len(v0) > 0
            and all(isinstance(x, (int | float)) for x in v0)
        )
        return {InputId(k): [float(x) for x in v] for k, v in data.items()}

    def save_embeddings(
        self,
        output_dir: str | Path,
        embeddings_by_input_index: dict[InputId, list[float]],
    ) -> None:
        """Save embeddings relative to the provided results root.

        This uses `self.embeddings_by_input_index_path` as the relative location.
        """
        abs_path = self.get_embeddings_abs_path(output_dir)
        self.save_embeddings_abs(abs_path, embeddings_by_input_index)

    def load_embeddings(self, output_dir: str | Path) -> dict[InputId, list[float]]:
        """Load embeddings using the relative path stored in this object."""
        abs_path = self.get_embeddings_abs_path(output_dir)
        return self.load_embeddings_abs(abs_path)
