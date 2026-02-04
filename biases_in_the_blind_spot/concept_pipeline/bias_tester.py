from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)
from biases_in_the_blind_spot.concept_pipeline.statistics import (
    calculate_statistics_single_group,
    calculate_statistics_two_groups,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair import VariationPair
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)


@dataclass
class BiasTester:
    responses_generator: ResponsesGenerator
    parse_response_fn: Callable[[InputId, str], int | None]
    variations_bias_temperature: float = 0.7
    n_responses_per_variation: int = 1
    variating_input_parameter: str = "resume"
    retry_on_empty_responses: int = 10

    class EmptyResponseError(RuntimeError):
        pass

    def _compute_statistics_for_result(
        self,
        result: ConceptBiasTestResult,
    ) -> None:
        """Compute and attach statistics for a ConceptBiasTestResult with positive/negative."""
        all_positive_acceptances: list[int | None] = []
        all_negative_acceptances: list[int | None] = []
        for per_input in result.responses_by_input.values():
            for pair_responses in per_input.values():
                all_positive_acceptances.extend(
                    pair_responses.positive_acceptances.values()
                )
                all_negative_acceptances.extend(
                    pair_responses.negative_acceptances.values()
                )

        result.statistics_positive_vs_negative = calculate_statistics_two_groups(
            all_positive_acceptances, all_negative_acceptances
        )
        result.statistics_positive = calculate_statistics_single_group(
            all_positive_acceptances
        )
        result.statistics_negative = calculate_statistics_single_group(
            all_negative_acceptances
        )

    def recompute_statistics(
        self,
        result: ConceptBiasTestResult,
    ) -> None:
        """Public wrapper to recompute statistics on an existing result."""
        self._compute_statistics_for_result(result)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "variations_bias_temperature": self.variations_bias_temperature,
            "n_responses_per_variation": self.n_responses_per_variation,
            "variating_input_parameter": self.variating_input_parameter,
            "retry_on_empty_responses": self.retry_on_empty_responses,
        }

    @staticmethod
    def _is_empty_response(resp: str) -> bool:
        return (not isinstance(resp, str)) or (len(resp.strip()) == 0)

    def _empty_response_indices(self, responses: list[str]) -> list[int]:
        return [i for i, resp in enumerate(responses) if self._is_empty_response(resp)]

    def calculate_acceptances_for_responses(
        self, input_id: InputId, responses: list[str]
    ) -> list[int | None]:
        assert isinstance(input_id, InputId)
        assert isinstance(responses, list) and len(responses) > 0, (
            "No responses provided for acceptance calculation"
        )
        return [self.parse_response_fn(input_id, resp) for resp in responses]

    def test_variatons_batch(
        self,
        input_template: str,
        input_parameters_by_input: dict[InputId, dict[str, str]],
        variations_by_concept: dict[
            ConceptId, dict[InputId, dict[VariationPairId, VariationPair]]
        ],
    ) -> dict[ConceptId, ConceptBiasTestResult]:
        """Batch test bias for multiple concepts at once.

        For each concept and each (input, pair) variation, generate n_responses_per_variation
        for both positive and negative variations, then compute acceptances and statistics.
        """
        assert self.n_responses_per_variation >= 1
        assert (
            isinstance(variations_by_concept, dict) and len(variations_by_concept) > 0
        )
        assert input_parameters_by_input is not None

        # Temporary storage per (concept_id, input_idx, pair_idx, side)
        temp_positive: dict[
            tuple[ConceptId, InputId, VariationPairId], dict[ResponseId, str]
        ] = {}
        temp_negative: dict[
            tuple[ConceptId, InputId, VariationPairId], dict[ResponseId, str]
        ] = {}
        batched_params: list[dict[str, str]] = []
        batched_meta: list[tuple[ConceptId, InputId, VariationPairId, str]] = []

        # Collect missing prompts across all concepts/inputs/pairs/sides
        for concept_id, variations_by_input in variations_by_concept.items():
            assert (
                isinstance(variations_by_input, dict) and len(variations_by_input) > 0
            )
            for input_id in sorted(variations_by_input.keys()):
                variation_pairs = variations_by_input.get(input_id, {})
                assert isinstance(variation_pairs, dict) and len(variation_pairs) >= 1
                for pair_id, variation_pair in variation_pairs.items():
                    assert isinstance(variation_pair, VariationPair)
                    temp_positive[(concept_id, input_id, pair_id)] = {}
                    temp_negative[(concept_id, input_id, pair_id)] = {}

                    missing_positive = self.n_responses_per_variation - len(
                        temp_positive[(concept_id, input_id, pair_id)]
                    )
                    missing_negative = self.n_responses_per_variation - len(
                        temp_negative[(concept_id, input_id, pair_id)]
                    )

                    for _ in range(missing_positive):
                        base_params = input_parameters_by_input.get(input_id)
                        assert base_params is not None
                        batched_params.append(
                            {
                                **base_params,
                                self.variating_input_parameter: variation_pair.positive,
                            }
                        )
                        batched_meta.append((concept_id, input_id, pair_id, "positive"))

                    for _ in range(missing_negative):
                        base_params = input_parameters_by_input.get(input_id)
                        assert base_params is not None
                        batched_params.append(
                            {
                                **base_params,
                                self.variating_input_parameter: variation_pair.negative,
                            }
                        )
                        batched_meta.append((concept_id, input_id, pair_id, "negative"))

        # Single generation call for all missing prompts
        if len(batched_params) > 0:
            new_resps_all = self.responses_generator.generate(
                input_template,
                batched_params,
                temperature=self.variations_bias_temperature,
            )
            assert len(new_resps_all) == len(batched_params)

            empty_indices = self._empty_response_indices(new_resps_all)
            attempts = 0
            while len(empty_indices) > 0 and attempts < self.retry_on_empty_responses:
                attempts += 1
                retry_params = [batched_params[i] for i in empty_indices]
                retry_resps = self.responses_generator.generate(
                    input_template,
                    retry_params,
                    temperature=self.variations_bias_temperature,
                )
                assert len(retry_resps) == len(retry_params)
                for idx, resp in zip(empty_indices, retry_resps, strict=True):
                    new_resps_all[idx] = resp
                empty_indices = self._empty_response_indices(new_resps_all)

            if len(empty_indices) > 0:
                failed_meta = [batched_meta[i] for i in empty_indices]
                raise self.EmptyResponseError(
                    f"Empty variation responses after {self.retry_on_empty_responses} retries: {failed_meta}"
                )

            for resp, (cid, iid, pid, side) in zip(
                new_resps_all, batched_meta, strict=True
            ):
                rid = ResponseId()
                if side == "positive":
                    while rid in temp_positive[(cid, iid, pid)]:
                        rid = ResponseId()
                    temp_positive[(cid, iid, pid)][rid] = resp
                else:
                    while rid in temp_negative[(cid, iid, pid)]:
                        rid = ResponseId()
                    temp_negative[(cid, iid, pid)][rid] = resp

        # Build per-concept results and compute acceptances
        out: dict[ConceptId, ConceptBiasTestResult] = {}
        for concept_id, variations_by_input in variations_by_concept.items():
            responses_by_input: dict[
                InputId, dict[VariationPairId, VariationPairResponses]
            ] = {}

            for input_id in sorted(variations_by_input.keys()):
                if input_id not in responses_by_input:
                    responses_by_input[input_id] = {}
                variation_pairs = variations_by_input.get(input_id, {})
                for pair_id, _variation_pair in variation_pairs.items():
                    effective_pair_id = pair_id
                    while effective_pair_id in responses_by_input[input_id]:
                        effective_pair_id = VariationPairId()

                    pos_responses = temp_positive[
                        (concept_id, input_id, effective_pair_id)
                    ]
                    neg_responses = temp_negative[
                        (concept_id, input_id, effective_pair_id)
                    ]
                    assert len(pos_responses) == self.n_responses_per_variation
                    assert len(neg_responses) == self.n_responses_per_variation

                    pos_acceptances = {
                        rid: self.parse_response_fn(input_id, resp)
                        for rid, resp in pos_responses.items()
                    }
                    neg_acceptances = {
                        rid: self.parse_response_fn(input_id, resp)
                        for rid, resp in neg_responses.items()
                    }

                    responses_by_input[input_id][effective_pair_id] = (
                        VariationPairResponses(
                            positive_responses=pos_responses,
                            negative_responses=neg_responses,
                            positive_acceptances=pos_acceptances,
                            negative_acceptances=neg_acceptances,
                        )
                    )

            result = ConceptBiasTestResult(
                responses_by_input=responses_by_input,
            )

            self._compute_statistics_for_result(result)

            out[concept_id] = result

        return out
