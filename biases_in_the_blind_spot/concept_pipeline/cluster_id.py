"""Core alias representing cluster identifiers within the concept pipeline."""


class ClusterId(int):
    """Lightweight int subclass used to denote cluster identifiers."""

    __slots__ = ()

    def __new__(cls, value: int | str) -> "ClusterId":
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"ClusterId expects int-compatible value, got {value!r}"
                ) from exc
        else:
            parsed = int(value)
        return int.__new__(cls, parsed)
