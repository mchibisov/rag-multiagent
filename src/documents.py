from pathlib import Path


def load_text_file(file_path: str) -> str:
    """Read a UTF-8 text file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """Group paragraphs into chunks without cutting paragraphs in half."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        separator = "\n\n" if current_chunk else ""
        candidate = f"{current_chunk}{separator}{paragraph}"

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
