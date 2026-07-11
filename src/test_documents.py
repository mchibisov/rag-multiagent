from documents import load_text_file, split_into_chunks

text = load_text_file("data/routepilot.txt")
chunks = split_into_chunks(text, chunk_size=500)

print(f"Длина документа: {len(text)} символов")
print(f"Количество фрагментов: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Фрагмент {index} ---")
    print(chunk)
