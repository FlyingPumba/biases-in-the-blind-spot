"""Core alias representing variation pair identifiers within the concept pipeline."""

from uuid import UUID, uuid4


class VariationPairId(str):
    """Lightweight string subclass used to denote variation pair identifiers."""

    __slots__ = ()

    def __new__(cls, value: str | UUID | None = None) -> "VariationPairId":
        if value is None:
            normalized = str(uuid4())
        else:
            normalized = str(value)
            try:
                UUID(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"VariationPairId expects UUID value, got {normalized!r}"
                ) from exc
        return str.__new__(cls, normalized)
