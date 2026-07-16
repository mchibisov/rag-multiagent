# Мультиагентный RAG с Mistral

[English version](README.md)

Небольшой учебный проект Retrieval-Augmented Generation без LangChain, LangGraph, FAISS и векторной базы данных. Он показывает основные механизмы напрямую на Python: разбиение документа, локальные embeddings, семантический поиск через NumPy, генерацию через Mistral API и последовательность из трёх LLM-ролей.

## Архитектура

```text
Вопрос пользователя
    ↓
Router Agent
    ↓
Semantic Retriever
    ↓
RAG Agent
    ↓
Checker Agent
    ↓
Финальный ответ
```

- **Router Agent** определяет, относится ли вопрос к локальной базе знаний.
- **Retriever** создаёт embedding вопроса и выбирает наиболее похожие фрагменты.
- **RAG Agent** отвечает только на основе найденного контекста.
- **Checker Agent** удаляет утверждения, которые не подтверждены контекстом.

Retriever — это обычный детерминированный Python-код, а не LLM-агент.

## Технологии

- Python 3.10+
- Mistral API
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- NumPy
- python-dotenv

## Структура проекта

```text
rag-multiagent/
├── data/
│   └── routepilot.txt
├── src/
│   ├── __init__.py
│   ├── agents.py
│   ├── documents.py
│   ├── pipeline.py
│   └── retrieval.py
├── .env.example
├── .gitignore
├── main.py
└── requirements.txt
```

## Установка

```bash
git clone https://github.com/mchibisov/rag-multiagent.git
cd rag-multiagent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Активация в Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Создай локальный файл окружения:

```bash
cp .env.example .env
```

Затем замени `your_mistral_api_key` в `.env` на свой ключ. Файл `.env` исключён из Git.

## Запуск

```bash
python main.py
```

Примеры вопросов:

```text
Сколько длится пробный период?
Сколько стоит базовый тариф?
Управляет ли RoutePilot автомобилями напрямую?
Кто является генеральным директором компании?
```

Для выхода введи `exit`.

## Как работает поиск

1. Документ делится на фрагменты по абзацам.
2. Локальная Sentence Transformer-модель превращает каждый фрагмент в нормализованный вектор.
3. Вопрос превращается в вектор той же моделью.
4. NumPy считает сходство с помощью матричного умножения.
5. Два наиболее похожих фрагмента передаются Mistral как контекст.

Поскольку векторы нормализованы, скалярное произведение совпадает с cosine similarity.

## Ограничения

- Используется один локальный текстовый документ.
- Embeddings строятся заново при каждом запуске.
- Линейный поиск NumPy подходит только для небольшого учебного корпуса.
- Router и Checker могут ошибаться, поскольку это LLM-вызовы.
- Защита от prompt injection ограничена и не является гарантией безопасности.
- Нет постоянного векторного индекса, веб-интерфейса и автоматической системы оценки.

## Почему проект специально сделан простым

Цель — показать каждый важный этап RAG без скрытия логики внутри фреймворков. В production-системе обычно добавляют постоянный индекс, логирование и мониторинг, автоматическую оценку, повторные попытки, строгую конфигурацию моделей, обработку входных данных и масштабируемый retrieval.
