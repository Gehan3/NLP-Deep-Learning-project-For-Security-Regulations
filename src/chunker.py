##لسا محددناش كام chunks هنقسم بيهم ال text
#add parent title in parser
#if self.current retrieve all text while you split the result into requirements-implement?

from __future__ import annotations
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

@dataclass(frozen=True)
class ChunkerConfig:
    input_path: Path = Path("data/cleaned/iso27002_cleaned.json")
    parents_output_path: Path = Path("data/chunks/iso27002_parents.json")
    children_output_path: Path = Path("data/chunks/iso27002_children.json")
    #change and try to see the results
    child_target_tokens: int = 66     
    overlap_tokens: int = 12         
    min_child_tokens: int = 8       
    tokenizer_name: str = "cl100k_base" # used for GPT-change according to our model
    #str = "meta-llama/Llama-3.2-3B-Instruct" 

    sections: tuple[str, ...] = ("control", "purpose", "guidance", "other_information")

    standard_name: str = "ISO 27002"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chunker")

class TokenCounter:
 
    def __init__(self, tokenizer_name: str = "cl100k_base") -> None:
        self._encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(tokenizer_name)
                logger.info("Tokenizer: tiktoken '%s' loaded.", tokenizer_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("tiktoken failed to load (%s); using fallback counter.", exc)
        else:
            logger.warning(
                "tiktoken not installed - using an approximate word-based "
                "token counter. Run `pip install tiktoken` for exact counts."
            )
 
    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, round(len(text.split()) * 1.3))
 
    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
    
        if self._encoder is not None:
            ids = self._encoder.encode(text)
            return [
                self._encoder.decode(ids[i:i + max_tokens])
                for i in range(0, len(ids), max_tokens)
            ]
        words = text.split()
        approx_words = max(1, int(max_tokens / 1.3))
        return [
            " ".join(words[i:i + approx_words])
            for i in range(0, len(words), approx_words)
        ]

# =======================================================================
class SentenceSplitter: #for protection
    _ABBREVIATIONS = (
        "e.g.", "i.e.", "etc.", "vs.", "fig.", "ver.", "no.", "approx.",
        "cf.", "al.", "eq.", "std.", "ed.", "rev.", "cl.", "sec.", "app.",
        "mr.", "mrs.", "ms.", "dr.", "jr.", "sr.",
    )
    _DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)") #lookbehind +look ahead

    _BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c\'(0-9])')

    _PLACEHOLDER = "\u0000DOT\u0000"

    def _protect(self, text: str) -> str:
        protected = self._DECIMAL_RE.sub(self._PLACEHOLDER, text)
        for abbr in self._ABBREVIATIONS:
            masked = abbr.replace(".", self._PLACEHOLDER)
            protected = re.sub(re.escape(abbr), masked, protected, flags=re.IGNORECASE)
        return protected

    def _restore(self, text: str) -> str:
        return text.replace(self._PLACEHOLDER, ".")

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        protected = self._protect(text.strip())
        raw_sentences = self._BOUNDARY_RE.split(protected)
        return [
            self._restore(s).strip()
            for s in raw_sentences
            if s and s.strip()
        ]
# Data containers
@dataclass
class ParentChunk:
    parent_id: str
    control_id: str
    title: str
    section: str
    text: str
    page: int

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "control_id": self.control_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "page": self.page,
        }


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    control_id: str
    section: str
    text: str
    token_count: int
    metadata_context: str

    def to_dict(self) -> dict:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "control_id": self.control_id,
            "section": self.section,
            "text": self.text,
            "token_count": self.token_count,
            "metadata_context": self.metadata_context,
        }

# Core Parent-Child Chunker

class ParentChildChunker:
    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config
        self.splitter = SentenceSplitter()
        self.tokens = TokenCounter(config.tokenizer_name)

        self.parents: list[ParentChunk] = []
        self.children: list[ChildChunk] = []

        self._skipped_sections = 0
        self._oversized_sentences = 0

    # ---- loading 

    def load_controls(self) -> list[dict]:
        path = self.config.input_path
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path.resolve()}. "
                "Run cleaner.py first to generate it."
            )
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

        if not isinstance(data, list) or not data:
            raise ValueError(f"{path} does not contain a non-empty list of controls.")

        return data

    # ---- metadata context ------------------------------------------

    def _build_metadata_context(self, control_id: str, title: str, section: str) -> str:
        return (
            f"{self.config.standard_name} Control {control_id}: {title} "
            f"- Section: {section.upper()}"
        )

    # ---- packing sentences into token-bounded, overlapping chunks --

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        cfg = self.config
        chunks: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.tokens.count(sentence)

            # Safety net: a single sentence bigger than the whole budget.
            if sent_tokens > cfg.child_target_tokens:
                if current:
                    chunks.append(current)
                    current, current_tokens = [], 0
                self._oversized_sentences += 1
                for piece in self.tokens.split_by_tokens(sentence, cfg.child_target_tokens):
                    chunks.append([piece])
                continue

            if current and current_tokens + sent_tokens > cfg.child_target_tokens:
                chunks.append(current)
                current, current_tokens = self._build_overlap(current)

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(current)

        return [" ".join(c) for c in chunks]

    def _build_overlap(self, previous_chunk: list[str]) -> tuple[list[str], int]:
        cfg = self.config
        overlap_sentences: list[str] = []
        overlap_tokens = 0
        for sentence in reversed(previous_chunk):
            t = self.tokens.count(sentence)
            if overlap_tokens + t > cfg.overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += t
        return overlap_sentences, overlap_tokens

    # ---- per-section child generation -------------------------------

    def _make_children_for_section(self, parent: ParentChunk) -> list[ChildChunk]:
        cfg = self.config
        sentences = self.splitter.split(parent.text)
        if not sentences:
            return []

        metadata_context = self._build_metadata_context(
            parent.control_id, parent.title, parent.section
        )

        raw_chunks = self._pack_sentences(sentences)

        children: list[ChildChunk] = []
        seq = 1
        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            n_tokens = self.tokens.count(chunk_text)

            # Merge tiny trailing crumbs into the previous child instead
            # of emitting a near-empty, low-signal embedding target.
            if n_tokens < cfg.min_child_tokens and children:
                children[-1].text = f"{children[-1].text} {chunk_text}".strip()
                children[-1].token_count = self.tokens.count(children[-1].text)
                continue

            child_id = f"child_{parent.control_id}_{parent.section}_{seq}"
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent.parent_id,
                    control_id=parent.control_id,
                    section=parent.section,
                    text=chunk_text,
                    token_count=n_tokens,
                    metadata_context=metadata_context,
                )
            )
            seq += 1

        return children





    # Main Pipeline

    def process(self) -> None:
        controls = self.load_controls()
        logger.info("Loaded controls from ", len(controls), self.config.input_path)

        for control in controls:
            control_id = str(control.get("control_id", "")).strip()
            title = str(control.get("title", "")).strip()
            page = control.get("page", 0)

            if not control_id:
                logger.warning("Skipping a control with missing control_id: %r", control)
                continue

            for section in self.config.sections:
                section_text = control.get(section, "")
                if not section_text or not str(section_text).strip():
                    self._skipped_sections += 1
                    continue

                section_text = str(section_text).strip()
                parent_id = f"{control_id}_{section}"

                parent = ParentChunk(
                    parent_id=parent_id,
                    control_id=control_id,
                    title=title,
                    section=section,
                    text=section_text,
                    page=int(page) if isinstance(page, (int, float)) else 0,
                )
                self.parents.append(parent)
                self.children.extend(self._make_children_for_section(parent))

        logger.info(
            "Processed %d controls -> %d parents / %d children "
            "(%d empty sections skipped, %d oversized sentences hard-split)",
            len(controls), len(self.parents), len(self.children),
            self._skipped_sections, self._oversized_sentences,
        )

    #save

    def save(self) -> None:
        output_dir = self.config.parents_output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        def write_json(path, data):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([item.to_dict() for item in data], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved -> {path}")
        write_json(self.config.parents_output_path, self.parents)
        write_json(self.config.children_output_path, self.children)

    # ---- stats -----------------------------------------------------------
    #نتاكد بيها الدنيا تمام وبعدها نشيلها
    def print_stats(self) -> None:
        if not self.children:
            logger.warning("No children were generated.")
            return

        token_counts = [c.token_count for c in self.children]
        avg_tokens = sum(token_counts) / len(token_counts)

        by_section: dict[str, int] = {}
        for c in self.children:
            by_section[c.section] = by_section.get(c.section, 0) + 1

        print("\n" + "=" * 60)
        print("CHUNKING STATISTICS")
        print("=" * 60)
        print(f"Total parents:            {len(self.parents)}")
        print(f"Total children:           {len(self.children)}")
        print(f"Avg tokens / child:       {avg_tokens:.1f}")
        print(f"Min / Max tokens / child: {min(token_counts)} / {max(token_counts)}")
        print(f"Empty sections skipped:   {self._skipped_sections}")
        print(f"Oversized sentences:      {self._oversized_sentences}")
        print("Children per section:")
        for section, count in sorted(by_section.items()):
            print(f"  - {section:<18} {count}")
        print("=" * 60 + "\n")


# =======================================================================
# Entrypoint
# =======================================================================

def main() -> None:
    config = ChunkerConfig()
    chunker = ParentChildChunker(config)

    try:
        chunker.process()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    chunker.save()
    chunker.print_stats()

    if chunker.children:
        print("Sample child chunk:\n")
        print(json.dumps(chunker.children[0].to_dict(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    
    config = ChunkerConfig(
        input_path=Path("data/cleaned/iso27002_cleaned.json")
    )
    chunker = ParentChildChunker(config)
    
    chunker.process()      
    chunker.save()         
    chunker.print_stats()
