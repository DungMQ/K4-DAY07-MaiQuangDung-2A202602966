from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Tách câu dựa trên ranh giới . ! ? theo sau là khoảng trắng hoặc xuống dòng
        # Dùng lookbehind để giữ lại dấu câu ở cuối mỗi câu
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            splits = list(current_text)
        elif sep in current_text:
            splits = current_text.split(sep)
        else:
            return self._split(current_text, next_seps)

        good_pieces: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good_pieces.append(piece)
            else:
                sub_chunks = self._split(piece, next_seps)
                good_pieces.extend(sub_chunks)

        if not good_pieces:
            return []

        join_sep = sep if (sep != "" and sep in current_text) else ""
        merged_chunks: list[str] = []
        current_chunk = ""

        for piece in good_pieces:
            if not current_chunk:
                current_chunk = piece
            else:
                candidate = current_chunk + join_sep + piece
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    merged_chunks.append(current_chunk)
                    current_chunk = piece

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks


class HeadingChunker:
    """
    Split text based on Markdown headings (# , ## , ### ).
    Long sections are further chunked using RecursiveChunker, while keeping the section heading.
    """

    def __init__(self, max_size: int = 400) -> None:
        self.max_size = max_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)
        chunks: list[str] = []

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            if len(sec_clean) <= self.max_size:
                chunks.append(sec_clean)
            else:
                lines = sec_clean.split("\n", 1)
                heading = lines[0].strip() if len(lines) > 0 else ""
                body = lines[1].strip() if len(lines) > 1 else sec_clean

                effective_chunk_size = max(100, self.max_size - len(heading) - 2)
                sub_chunker = RecursiveChunker(chunk_size=effective_chunk_size)
                sub_chunks = sub_chunker.chunk(body)

                if not sub_chunks:
                    chunks.append(sec_clean[: self.max_size])
                else:
                    for sc in sub_chunks:
                        if heading and not sc.startswith(heading):
                            chunks.append(f"{heading}\n{sc}")
                        else:
                            chunks.append(sc)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot_prod / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return result
