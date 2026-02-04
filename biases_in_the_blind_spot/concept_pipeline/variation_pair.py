from dataclasses import dataclass

from dataclass_wizard import JSONWizard


@dataclass
class VariationPair(JSONWizard):
    positive: str
    negative: str

    def get_variation_by_side(self, side: str) -> str:
        if side == "positive":
            return self.positive
        elif side == "negative":
            return self.negative
        else:
            raise ValueError(f"Unknown side: {side}")
