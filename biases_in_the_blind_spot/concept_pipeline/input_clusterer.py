from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.input_clusters import InputClusters
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId


@dataclass
class InputClusterer:
    """Embeds sanitized inputs using OpenAI embeddings, clusters via k-means, picks representatives.

    Assumptions:
    - OpenAI Python client is configured via environment (OPENAI_API_KEY).
    - `dataset.sanitized_varying_inputs` is populated.
    - Number of clusters equals `dataset.n_representative_input_clusters`.
    - Representatives are chosen by minimum L2 distance to cluster centroid.
    """

    embedding_model: str = "text-embedding-3-large"

    @property
    def config(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
        }

    async def _embed_batch(
        self, texts: dict[InputId, str], max_concurrent: int = 100
    ) -> dict[InputId, list[float]]:
        print(
            f"Embedding {len(texts)} texts using {self.embedding_model} (max_concurrent={max_concurrent})"
        )
        assert isinstance(max_concurrent, int) and max_concurrent >= 1
        aclient = AsyncOpenAI()
        semaphore = asyncio.Semaphore(max_concurrent)

        keys_sorted = sorted(texts.keys())
        total = len(keys_sorted)
        pbar = tqdm(total=total, desc="Embedding texts")

        async def embed_one(idx: InputId) -> tuple[InputId, list[float]]:
            async with semaphore:
                text = texts[idx].replace("\n", " ")
                resp = await aclient.embeddings.create(
                    input=[text], model=self.embedding_model
                )
                vec = resp.data[0].embedding
                assert isinstance(vec, list) and len(vec) > 0
                out_vec = [float(x) for x in vec]
                pbar.update(1)
                return idx, out_vec

        tasks = [embed_one(i) for i in keys_sorted]
        datasets = await asyncio.gather(*tasks)
        pbar.close()

        out: dict[InputId, list[float]] = dict(datasets)
        assert len(out) == len(texts)
        return out

    @staticmethod
    def _kmeans_cluster(
        embeddings_by_idx: dict[InputId, list[float]], n_clusters: int
    ) -> tuple[list[list[InputId]], list[InputId]]:
        print(
            f"Clustering {len(embeddings_by_idx)} embeddings into {n_clusters} clusters"
        )
        assert isinstance(n_clusters, int) and n_clusters >= 1
        idxs = sorted(embeddings_by_idx.keys())
        vectors = np.array([embeddings_by_idx[i] for i in idxs], dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[0] == len(idxs)
        n_clusters = min(n_clusters, len(idxs))
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        labels = km.fit_predict(vectors)

        clusters: list[list[InputId]] = [[] for _ in range(n_clusters)]
        for i, lbl in enumerate(labels):
            clusters[int(lbl)].append(idxs[i])

        # Representatives: closest to centroid by L2
        representatives: list[InputId] = []
        for c_id in range(n_clusters):
            members = clusters[c_id]
            assert len(members) > 0
            member_vecs = np.array(
                [embeddings_by_idx[m] for m in members], dtype=np.float32
            )
            centroid = member_vecs.mean(axis=0)
            dists = ((member_vecs - centroid) ** 2).sum(axis=1)
            best_idx = int(np.argmin(dists))
            representatives.append(members[best_idx])
        assert len(representatives) == len(clusters)
        return clusters, representatives

    async def cluster_inputs_if_needed(
        self, dataset: ConceptPipelineDataset, output_dir: str | Path
    ) -> None:
        assert dataset.sanitized_varying_inputs is not None
        assert dataset.n_representative_input_clusters is not None

        if dataset.input_clusters is not None:
            # Check if embeddings file actually exists
            embeddings_path = dataset.input_clusters.get_embeddings_abs_path(output_dir)
            if embeddings_path.exists():
                return
            else:
                # Clusters exist but embeddings file is missing - regenerate only embeddings
                print(
                    f"Input clusters exist but embeddings file missing at {embeddings_path}."
                )
                print("Regenerating embeddings only (keeping existing clusters)...")
                embeddings_by_idx = await self._embed_batch(
                    dataset.sanitized_varying_inputs
                )
                dataset.input_clusters.save_embeddings(output_dir, embeddings_by_idx)
                print("Embeddings regenerated successfully")
                return

        # No clusters exist yet - do full clustering
        print("No input clusters found. Running full clustering...")
        embeddings_by_idx = await self._embed_batch(dataset.sanitized_varying_inputs)
        clusters, representatives = self._kmeans_cluster(
            embeddings_by_idx, dataset.n_representative_input_clusters
        )
        # Persist embeddings to a pickle at the datasets root. We include dataset_name
        # in the filename to avoid collisions across runs in the same output_dir.
        assert isinstance(dataset.dataset_name, str) and len(dataset.dataset_name) > 0
        input_clusters = InputClusters(
            embeddings_by_input_index_path=f"embeddings_{dataset.dataset_name}.pkl",
            clusters=clusters,
            representatives=representatives,
        )
        input_clusters.save_embeddings(output_dir, embeddings_by_idx)
        dataset.input_clusters = input_clusters

    def export_clusters_html(
        self, figures_root_directory: str | Path, dataset: ConceptPipelineDataset
    ) -> None:
        """Export a single HTML summarizing input clusters and representatives.

        Assumptions:
        - `dataset.input_clusters` is set
        - `dataset.sanitized_varying_inputs` contains texts for input indices
        - `figures_root_directory` exists or can be created
        """
        assert dataset.input_clusters is not None
        assert dataset.sanitized_varying_inputs is not None
        root = Path(str(figures_root_directory))
        root.mkdir(parents=True, exist_ok=True)

        clusters = dataset.input_clusters.clusters
        reps = dataset.input_clusters.representatives
        assert isinstance(clusters, list) and isinstance(reps, list)
        assert len(reps) == len(clusters)

        parts: list[str] = []
        parts.append("<h1>Input Clusters</h1>")
        parts.append(f"<p>Num clusters: {len(clusters)}</p>")

        for cidx, members in enumerate(clusters):
            assert isinstance(members, list) and len(members) > 0
            rep = reps[cidx]
            assert rep in members
            parts.append(f"<h2>Cluster {cidx + 1} (size {len(members)})</h2>")
            parts.append(
                f"<p>Members: {', '.join(str(i) for i in sorted(members))}</p>"
            )
            rep_text = dataset.sanitized_varying_inputs[rep]
            assert isinstance(rep_text, str) and len(rep_text) > 0
            parts.append(f"<h3>Representative input (index {rep})</h3>")
            parts.append(
                '<pre style="white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;">'
                + rep_text
                + "</pre>"
            )

        html = "\n\n".join(parts)
        out_path = root / "input_clusters.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
