"""Core type alias representing concept identifiers."""

from uuid import UUID, uuid4


class ConceptId(str):
    """Lightweight string subclass used to denote concept identifiers."""

    __slots__ = ()

    def __new__(cls, value: str | UUID | None = None) -> "ConceptId":
        if value is None:
            normalized = str(uuid4())
        else:
            normalized = str(value)
            try:
                UUID(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"ConceptId expects UUID value, got {normalized!r}"
                ) from exc
        return str.__new__(cls, normalized)
