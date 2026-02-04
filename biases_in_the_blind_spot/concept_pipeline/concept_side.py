"""Utility helpers for concept sides."""

from typing import Literal

from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId

Side = Literal["positive", "negative"]


class ConceptSide(str):
    """
    String subclass representing a concept side, stored as 'concept_id:side'.
    Can be used as a dictionary key and serializes cleanly to JSON as a string.
    """

    __slots__ = ()
    SEP = ":"
    VALID: set[str] = {"positive", "negative"}

    def __new__(cls, concept_id: ConceptId | str | dict, side: Side | None = None):
        """
        Usage:
          ConceptSide(123, "positive")  -> "123:positive"
          ConceptSide("123:positive")   -> "123:positive" (validated & normalized)
          ConceptSide({"concept_id": 123, "side": "positive"})
            -> "123:positive" (backward compatibility)
        """
        if isinstance(concept_id, dict):
            assert side is None, "Cannot provide side when concept_id is a dict"
            cid = concept_id["concept_id"]
            side_str = concept_id["side"]
            normalized_cid = ConceptId(cid)
            assert side_str in cls.VALID, (
                f"side must be one of {cls.VALID}, got {side_str!r}"
            )
            s_norm = f"{normalized_cid}{cls.SEP}{side_str}"
            return str.__new__(cls, s_norm)

        if side is None:
            s = str(concept_id)
            try:
                cid_str, side_str = s.split(cls.SEP, 1)
            except ValueError as exc:
                raise ValueError(
                    f"Expected 'concept_id{cls.SEP}side', got {s!r}"
                ) from exc
            try:
                cid = ConceptId(cid_str)
            except ValueError as exc:
                raise ValueError(f"concept_id must be int, got {cid_str!r}") from exc
            if side_str not in cls.VALID:
                raise ValueError(f"side must be one of {cls.VALID}, got {side_str!r}")
            s_norm = f"{cid}{cls.SEP}{side_str}"
            return str.__new__(cls, s_norm)

        assert isinstance(side, str)
        try:
            cid = ConceptId(concept_id)
        except ValueError as exc:
            raise ValueError(f"concept_id must be int, got {concept_id!r}") from exc
        s_norm = f"{cid}{cls.SEP}{side}"
        return str.__new__(cls, s_norm)

    @property
    def concept_id(self) -> ConceptId:
        return ConceptId(self.split(self.SEP, 1)[0])

    @property
    def side(self) -> Side:
        return self.split(self.SEP, 1)[1]  # type: ignore[return-value]

    @classmethod
    def from_str(cls, s: str) -> "ConceptSide":
        return cls(s)
