from dataclasses import dataclass

from dataclass_wizard import JSONWizard

from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)


@dataclass
class ConceptBiasTestResult(JSONWizard):
    """Side-specific variation-bias test results without flattening across inputs.

    responses_by_input:
        input_index -> pair_index -> [responses_for_this_side]
    acceptances_by_input:
        Same keys, values are 0/1/None parsed acceptances aligned to responses.

    Index semantics:
    - input_id: index of the sanitized varying input under test,
      i.e., the key into ConceptVariations.variations and into
      ConceptPipelineResult.sanitized_varying_inputs.
    - pair_id: index within the list of variations for this side at
      the given input_id.
    """

    responses_by_input: dict[
        InputId, dict[VariationPairId, VariationPairResponses]
    ]  # input_index -> pair_index -> [responses for each variation pair]

    # Derived: input_index -> pair_index -> whether acceptances "flip" between positive and negative
    # (requires exactly one response per side per pair; see variation_bias._compute_flipped_variation_pairs).
    flipped_variation_pairs: dict[InputId, dict[VariationPairId, bool]] | None = None

    # Statistics computed for this side only
    statistics_positive_vs_negative: dict[str, float | None] | None = None
    statistics_positive: dict[str, float] | None = None
    statistics_negative: dict[str, float] | None = None

    @property
    def num_variations(self) -> int:
        total = 0
        for per_input in self.responses_by_input.values():
            total += len(per_input)
        return total

    def flatten_acceptances(self) -> list[int | None]:
        """Return flattened acceptances for this concept side across all inputs/pairs."""
        flat: list[int | None] = []
        for input_index in sorted(self.responses_by_input.keys()):
            per_pairs = self.responses_by_input[input_index]
            for pair_index in sorted(per_pairs.keys()):
                pair_responses = per_pairs[pair_index]
                flat.extend(pair_responses.positive_acceptances.values())
                flat.extend(pair_responses.negative_acceptances.values())
        return flat
