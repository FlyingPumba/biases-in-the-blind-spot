"""Core alias representing response identifiers within the concept pipeline."""

from uuid import UUID, uuid4


class ResponseId(str):
    """Lightweight string subclass used to denote response identifiers."""

    __slots__ = ()

    def __new__(cls, value: str | UUID | None = None) -> "ResponseId":
        if value is None:
            normalized = str(uuid4())
        else:
            normalized = str(value)
            try:
                UUID(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"ResponseId expects UUID value, got {normalized!r}"
                ) from exc
        return str.__new__(cls, normalized)
