"""Read indexed and plain-text detection labels without shifting class IDs."""

from pathlib import Path


def parse_labels(text: str) -> list[str]:
    entries: dict[int, str] = {}
    plain: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        number, sep, name = line.partition(" ")
        if sep and number.isdigit():
            entries[int(number)] = name.strip()
        else:
            plain.append(line)
    if entries and plain:
        raise ValueError("Cannot mix indexed and plain label formats")
    if entries:
        if max(entries) > 10000:
            raise ValueError("Label index too large")
        return [entries.get(index, f"class_{index}") for index in range(max(entries) + 1)]
    return plain


def read_labels(path: Path) -> list[str]:
    return parse_labels(path.read_text(encoding="utf-8"))
