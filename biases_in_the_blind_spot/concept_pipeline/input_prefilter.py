from dataclasses import dataclass

import numpy as np

from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId


@dataclass
class InputPrefilter:
    max_inputs: int | None = None

    @property
    def config(self) -> dict[str, int | None]:
        return {
            "max_inputs": self.max_inputs,
        }

    def filter_inputs(
        self,
        input_ids: list[InputId],
        baseline_acceptances_by_input: dict[InputId, dict[ResponseId, int | None]],
    ) -> list[InputId]:
        """Filter inputs based on variance in their baseline acceptance responses.

        Args:
            input_ids: List of input IDs to consider for filtering
            baseline_acceptances_by_input: Map of input ID -> response ID -> acceptance value

        Returns:
            Filtered list of input IDs sorted by variance (highest first), limited to max_inputs
        """
        if len(input_ids) == 0:
            return []

        if self.max_inputs is not None and len(input_ids) <= self.max_inputs:
            return input_ids

        input_variances: list[tuple[InputId, float]] = []
        num_nonzero_variance = 0
        for input_id in input_ids:
            acceptances_map = baseline_acceptances_by_input.get(input_id, {})
            acceptances = list(acceptances_map.values())
            valid_acceptances = [a for a in acceptances if a is not None]

            if len(valid_acceptances) == 0:
                variance = 0.0
            else:
                variance = float(np.var(valid_acceptances))
            if variance > 0.0:
                num_nonzero_variance += 1

            input_variances.append((input_id, variance))

        print(
            f"total non-zero input variance: {num_nonzero_variance} / {len(input_ids)} = {num_nonzero_variance / len(input_ids):.2%}"
        )
        input_variances.sort(key=lambda x: x[1], reverse=True)

        filtered_input_ids = [input_id for input_id, _ in input_variances]
        if self.max_inputs is not None:
            filtered_input_ids = filtered_input_ids[: self.max_inputs]

        return filtered_input_ids
