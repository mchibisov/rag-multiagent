from .agents import checker_agent, rag_agent, router_agent
from .documents import load_text_file, split_into_chunks
from .retrieval import Retriever


class RAGPipeline:
    """Router -> retrieval -> answer generation -> verification."""

    def __init__(self, document_path: str):
        text = load_text_file(document_path)
        chunks = split_into_chunks(text, chunk_size=500)
        self.retriever = Retriever(chunks)

    def run(self, question: str) -> str:
        if not router_agent(question):
            return "Вопрос не относится к локальным документам."

        search_results = self.retriever.search(question, top_k=2)
        draft_answer = rag_agent(question, search_results)

        return checker_agent(
            question,
            draft_answer,
            search_results,
        )
