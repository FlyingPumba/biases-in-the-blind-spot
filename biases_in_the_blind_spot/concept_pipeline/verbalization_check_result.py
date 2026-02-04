from dataclasses import dataclass


@dataclass
class VerbalizationCheckResult:
    verbalized: bool
    witness: str
