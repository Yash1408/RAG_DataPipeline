"""Layout-aware PDF parser (text + tables + Item/Note context per page)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Iterator

import pdfplumber

_ITEM_RE = re.compile(r"^Item\s+\d+[A-Z]?\.", re.MULTILINE)
_NOTE_RE = re.compile(r"Note\s+\d+\s+[—-]\s+[A-Z][A-Z\s,&]+", re.MULTILINE)


@dataclass
class TableBlock:
    name: str
    header: list[str]
    rows: list[list[str]]
    page_no: int

    def cells(self) -> Iterator[tuple[str, str, str]]:
        for row in self.rows:
            if not row:
                continue
            row_label, *vals = row
            for col, val in zip(self.header[1:], vals):
                yield row_label, col, val


@dataclass
class ParsedPage:
    page_no: int
    text: str
    tables: list[TableBlock] = field(default_factory=list)
    item_title: str | None = None
    note_title: str | None = None
    section_title: str | None = None


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    metadata: dict


def parse_pdf(pdf_bytes: bytes) -> ParsedDocument:
    pages: list[ParsedPage] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        current_item = None
        current_note = None
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if m := _ITEM_RE.search(text):
                end = text.find("\n", m.start())
                current_item = text[m.start():end].strip()
            if m := _NOTE_RE.search(text):
                current_note = m.group(0).strip()
            tables: list[TableBlock] = []
            for raw in page.extract_tables() or []:
                if not raw or not raw[0]:
                    continue
                header = [c or "" for c in raw[0]]
                rows = [[c or "" for c in r] for r in raw[1:]]
                tables.append(TableBlock(
                    name=_infer_table_name(text, header),
                    header=header, rows=rows, page_no=i + 1,
                ))
            pages.append(ParsedPage(
                page_no=i + 1, text=text, tables=tables,
                item_title=current_item, note_title=current_note,
                section_title=_extract_section(text),
            ))
    return ParsedDocument(pages=pages, metadata={"total_pages": len(pages)})


def _extract_section(text: str) -> str | None:
    for ln in text.split("\n")[:10]:
        s = ln.strip()
        if 3 < len(s) < 80 and s.istitle() and not s.endswith("."):
            return s
    return None


def _infer_table_name(page_text: str, header: list[str]) -> str:
    first = header[0] if header and header[0] else ""
    if not first:
        return "unnamed_table"
    idx = page_text.find(first)
    if idx <= 0:
        return "unnamed_table"
    pre = page_text[:idx].rstrip().split("\n")
    for line in reversed(pre[-5:]):
        if line.strip().endswith(":") or "as follows" in line:
            return line.strip().rstrip(":")
    return pre[-1].strip() if pre else "unnamed_table"
