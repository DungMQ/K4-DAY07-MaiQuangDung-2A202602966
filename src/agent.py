from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> str:
        if metadata_filter:
            results = self.store.search_with_filter(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        else:
            results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks = []
        for idx, item in enumerate(results, start=1):
            source = (
                item.get("metadata", {}).get("source")
                or item.get("metadata", {}).get("doc_id")
                or item.get("id")
                or "unknown"
            )
            content = item.get("content", "").strip()
            context_blocks.append(f"[{idx}] (Nguồn: {source}):\n{content}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Bạn là một trợ lý AI thông minh dựa trên cơ sở tri thức.\n"
            f"Hãy trả lời câu hỏi dưới đây chỉ dựa vào ngữ cảnh được cung cấp. Trích dẫn số thứ tự nguồn [1], [2] tương ứng nếu có.\n"
            f"Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy thông tin.\n\n"
            f"--- Ngữ cảnh ---\n{context_str}\n\n"
            f"--- Câu hỏi ---\n{question}\n\n"
            f"--- Câu trả lời ---"
        )
        return self.llm_fn(prompt)

    def answer_with_filter(
        self,
        question: str,
        metadata_filter: dict | None = None,
        top_k: int = 3,
    ) -> str:
        """Trả lời câu hỏi kết hợp bộ lọc metadata."""
        return self.answer(question, top_k=top_k, metadata_filter=metadata_filter)
