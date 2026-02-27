from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
SKIP_TAGS = {"script", "style", "noscript"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTENT_TAGS = {"p", "li", "blockquote", "pre", "code", "td", "th", "section", "article", "main", "div"}


def normalize_vector(vec: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(value * value for value in vec))
    if magnitude == 0.0:
        return vec
    return [value / magnitude for value in vec]


class SectionExtractor(HTMLParser):
    """Extract page links and chunkable sections based on heading boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self._links: List[str] = []
        self._tag_stack: List[str] = []
        self._current_heading: str = ""
        self._current_lines: List[str] = []
        self._sections: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._tag_stack and self._tag_stack[-1] in SKIP_TAGS:
            return
        text = " ".join(data.split())
        if not text:
            return

        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag in HEADING_TAGS:
            self._flush_section()
            self._current_heading = text
            return

        if current_tag in CONTENT_TAGS:
            self._current_lines.append(text)

    def _flush_section(self) -> None:
        body = " ".join(self._current_lines).strip()
        if body:
            heading = self._current_heading or "Overview"
            self._sections.append((heading, body))
        self._current_lines = []

    @property
    def sections(self) -> List[Tuple[str, str]]:
        self._flush_section()
        return self._sections

    @property
    def links(self) -> List[str]:
        return self._links


@dataclass
class PageDocument:
    url: str
    title: str
    content: str
    section: str = "Overview"
    tags: List[str] = field(default_factory=list)


class MiniContextModel:
    """A lightweight semantic retriever using local sentence embeddings."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self.documents: List[PageDocument] = []
        self.doc_vectors: List[List[float]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in WORD_PATTERN.findall(text)]

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _token_embedding(self, token: str) -> List[float]:
        values: List[float] = []
        for idx in range(self.embedding_dim):
            digest = hashlib.blake2b(f"{token}:{idx}".encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, byteorder="big", signed=False)
            values.append((number / (2**64 - 1)) * 2 - 1)
        return values

    def embed_text(self, text: str) -> List[float]:
        tokens = self.tokenize(text)
        if not tokens:
            return [0.0] * self.embedding_dim

        accumulator = [0.0] * self.embedding_dim
        for token in tokens:
            token_vec = self._token_embedding(token)
            for idx, value in enumerate(token_vec):
                accumulator[idx] += value
        averaged = [value / len(tokens) for value in accumulator]
        return normalize_vector(averaged)

    def fit(self, documents: Iterable[PageDocument]) -> None:
        self.documents = list(documents)
        self.doc_vectors = [
            self.embed_text(f"{doc.title} {doc.section} {doc.content}") for doc in self.documents
        ]

    def search(
        self, query: str, top_k: int = 5, tags: Sequence[str] | None = None
    ) -> List[Tuple[PageDocument, float]]:
        if not self.documents:
            raise ValueError("Model is empty. Fit or load an index first.")

        query_vec = self.embed_text(query)
        tag_filter = {tag.lower() for tag in (tags or [])}

        scores = []
        for doc, doc_vec in zip(self.documents, self.doc_vectors):
            if tag_filter and not tag_filter.intersection({tag.lower() for tag in doc.tags}):
                continue
            score = self._cosine_similarity(doc_vec, query_vec)
            scores.append((doc, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> Dict[str, object]:
        return {
            "embedding_dim": self.embedding_dim,
            "documents": [asdict(doc) for doc in self.documents],
            "doc_vectors": self.doc_vectors,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "MiniContextModel":
        embedding_dim = int(payload.get("embedding_dim", 128))
        model = cls(embedding_dim=embedding_dim)
        model.documents = [PageDocument(**doc) for doc in payload["documents"]]
        model.doc_vectors = [[float(x) for x in vec] for vec in payload["doc_vectors"]]
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MiniContextModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def infer_tags(url: str) -> List[str]:
    lowered = url.lower()
    tags = []
    for candidate in ("docs", "blog", "changelog"):
        if f"/{candidate}" in lowered or lowered.endswith(candidate):
            tags.append(candidate)
    return tags


def crawl_site(base_url: str, max_pages: int = 20) -> List[PageDocument]:
    parsed_base = urlparse(base_url)
    if not parsed_base.scheme:
        raise ValueError("Base URL must include http:// or https://")

    queue = [base_url]
    seen = set(queue)
    docs: List[PageDocument] = []

    while queue and len(docs) < max_pages:
        url = queue.pop(0)
        try:
            with urlopen(url, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            continue

        parser = SectionExtractor()
        parser.feed(html)
        sections = parser.sections
        if not sections:
            continue

        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url
        page_tags = infer_tags(url)
        for section_title, section_text in sections:
            docs.append(
                PageDocument(
                    url=url,
                    title=title,
                    section=section_title,
                    content=section_text,
                    tags=page_tags,
                )
            )
            if len(docs) >= max_pages:
                break

        for link in parser.links:
            absolute = urljoin(url, link)
            parsed = urlparse(absolute)
            if parsed.netloc != parsed_base.netloc:
                continue
            normalized = parsed._replace(fragment="", query="").geturl().rstrip("/")
            if normalized and normalized not in seen:
                seen.add(normalized)
                queue.append(normalized)

    return docs


def cli() -> None:
    parser = argparse.ArgumentParser(description="Mini context-aware website search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Crawl a website and build index")
    build_parser.add_argument("base_url", help="Start URL to crawl")
    build_parser.add_argument("--max-pages", type=int, default=20)
    build_parser.add_argument("--out", default="context_index.json")

    search_parser = subparsers.add_parser("search", help="Search in a saved index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--index", default="context_index.json")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated metadata filters (docs,blog,changelog)",
    )

    args = parser.parse_args()

    if args.command == "build":
        docs = crawl_site(args.base_url, max_pages=args.max_pages)
        model = MiniContextModel()
        model.fit(docs)
        model.save(args.out)
        print(f"Indexed {len(docs)} sections into {args.out}")
    else:
        model = MiniContextModel.load(args.index)
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        for doc, score in model.search(args.query, top_k=args.top_k, tags=tags):
            print(
                f"[{score:.3f}] {doc.title} ({doc.section}) [{','.join(doc.tags) or 'untagged'}] -> {doc.url}"
            )


if __name__ == "__main__":
    cli()
