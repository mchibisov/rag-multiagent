from src.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline("data/routepilot.txt")

    print("RAG-система запущена. Для выхода введи: exit")

    while True:
        question = input("\nВопрос: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = pipeline.run(question)
        except Exception as error:
            print(f"Ошибка: {error}")
            continue

        print(f"Ответ: {answer}")


if __name__ == "__main__":
    main()
