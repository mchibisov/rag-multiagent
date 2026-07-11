from pathlib import Path


def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
