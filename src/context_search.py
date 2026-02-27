from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
SKIP_TAGS = {"script", "style", "noscript"}


class TextExtractor(HTMLParser):
    """Extract visible text and links from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: List[str] = []
        self._links: List[str] = []
        self._tag_stack: List[str] = []

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
        text = data.strip()
        if text:
            self._text_parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self._text_parts)

    @property
    def links(self) -> List[str]:
        return self._links


@dataclass
class PageDocument:
    url: str
    title: str
    content: str


class MiniContextModel:
    """A lightweight TF-IDF retriever for small websites."""

    def __init__(self) -> None:
        self.documents: List[PageDocument] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
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

    def fit(self, documents: Iterable[PageDocument]) -> None:
        self.documents = list(documents)
        term_freqs: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}

        for doc in self.documents:
            tf: Dict[str, int] = {}
            terms = self.tokenize(f"{doc.title} {doc.content}")
            for token in terms:
                tf[token] = tf.get(token, 0) + 1
            term_freqs.append(tf)
            for token in tf:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.vocabulary = {token: i for i, token in enumerate(sorted(doc_freq))}
        total_docs = max(len(self.documents), 1)
        self.idf = [0.0] * len(self.vocabulary)

        for token, idx in self.vocabulary.items():
            df = doc_freq[token]
            self.idf[idx] = math.log((1 + total_docs) / (1 + df)) + 1.0

        self.doc_vectors = [self._tfidf_vector(tf) for tf in term_freqs]

    def _tfidf_vector(self, term_freq: Dict[str, int]) -> List[float]:
        vec = [0.0] * len(self.vocabulary)
        total_terms = sum(term_freq.values()) or 1
        for token, count in term_freq.items():
            idx = self.vocabulary.get(token)
            if idx is None:
                continue
            tf = count / total_terms
            vec[idx] = tf * self.idf[idx]
        return vec

    def search(self, query: str, top_k: int = 5) -> List[Tuple[PageDocument, float]]:
        if not self.documents or not self.vocabulary:
            raise ValueError("Model is empty. Fit or load an index first.")

        query_tf: Dict[str, int] = {}
        for token in self.tokenize(query):
            query_tf[token] = query_tf.get(token, 0) + 1

        query_vec = self._tfidf_vector(query_tf)
        scores = [
            (doc, self._cosine_similarity(doc_vec, query_vec))
            for doc, doc_vec in zip(self.documents, self.doc_vectors)
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> Dict[str, object]:
        return {
            "documents": [asdict(doc) for doc in self.documents],
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "MiniContextModel":
        model = cls()
        model.documents = [PageDocument(**doc) for doc in payload["documents"]]
        model.vocabulary = {str(k): int(v) for k, v in payload["vocabulary"].items()}
        model.idf = [float(v) for v in payload["idf"]]
        model.doc_vectors = [[float(x) for x in vec] for vec in payload["doc_vectors"]]
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MiniContextModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


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

        parser = TextExtractor()
        parser.feed(html)
        text = parser.text
        if not text:
            continue

        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url
        docs.append(PageDocument(url=url, title=title, content=text))

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

    args = parser.parse_args()

    if args.command == "build":
        docs = crawl_site(args.base_url, max_pages=args.max_pages)
        model = MiniContextModel()
        model.fit(docs)
        model.save(args.out)
        print(f"Indexed {len(docs)} pages into {args.out}")
    else:
        model = MiniContextModel.load(args.index)
        for doc, score in model.search(args.query, top_k=args.top_k):
            print(f"[{score:.3f}] {doc.title} -> {doc.url}")


if __name__ == "__main__":
    cli()
