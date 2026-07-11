from sentence_transformers import SentenceTransformer

from documents import load_text_file, split_into_chunks


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

text = load_text_file("data/routepilot.txt")
chunks = split_into_chunks(text, chunk_size=500)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
)

print(f"Количество фрагментов: {len(chunks)}")
print(f"Форма массива embeddings: {embeddings.shape}")
print(f"Размер одного embedding: {len(embeddings[0])}")

print("\nПервые 10 чисел первого embedding:")
print(embeddings[0][:10])
