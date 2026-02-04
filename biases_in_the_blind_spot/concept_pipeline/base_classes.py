from collections.abc import Callable

ResultParser = Callable[[str], tuple[str, ...]]
