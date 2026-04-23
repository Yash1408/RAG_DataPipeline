"""
Generic verbatim-extraction primitives.

These functions describe *structure*, not *values*. They take a page of text
and pull out whatever happens to be there — so the extractor works on the
2024 10-K, the 2025 10-K, next year's 10-K, or any filing with a similar
table/narrative shape. No literal amounts, counts, or life-spans are
hard-coded anywhere.

Three primitives, in order of increasing complexity:

  extract_row_rightmost_money  — find a row label in a table-ish block, return
                                 the rightmost dollar amount on that row
                                 (10-K tables put the most-recent fiscal year
                                 on the right).
  extract_int_before           — pull the integer that appears before an
                                 anchor phrase, preserving the original
                                 surrounding words verbatim.
  extract_row_value_by_label   — find a labeled row in a "label ... value"
                                 layout (e.g. "Other equipment   Three to
                                 ten years") and return the value verbatim.

All functions return `None` when nothing matches — the verbatim gate then
drops the field rather than fabricating an answer.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
# Money token like "$ 128,725", "$128,725", "$ 1,234.5" — the whitespace after
# the dollar sign is optional because pdfplumber occasionally splits them.
_MONEY_RE = re.compile(r"\$\s*[\d,]+(?:\.\d+)?")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


# --------------------------------------------------------------------------- #
# Primitive 1: rightmost money on a row containing a label                    #
# --------------------------------------------------------------------------- #
def extract_row_rightmost_money(
    page_text: str,
    row_label: str,
    must_follow: str | None = None,
) -> str | None:
    """
    Find the FIRST line on `page_text` that:
      * contains `row_label` (e.g. "Net sales"),
      * optionally occurs AFTER a section marker (e.g. "AWS" segment block),
    and return the RIGHTMOST dollar-amount token on that line, verbatim.

    10-K segment tables list fiscal years left-to-right with the most-recent
    year on the right — so "rightmost" == "most recent". No year literal is
    required; the extractor naturally picks up whichever year is printed.
    """
    # If a section marker is provided, anchor the search after it (e.g. only
    # look at lines that come after the "AWS" segment header).
    search_text = page_text
    if must_follow:
        anchor = page_text.find(must_follow)
        if anchor == -1:
            return None
        search_text = page_text[anchor:]

    for line in search_text.splitlines():
        if row_label not in line:
            continue
        monies = _MONEY_RE.findall(line)
        if not monies:
            continue
        # Preserve the token exactly as it appeared on the page.
        return monies[-1].strip()
    return None


# --------------------------------------------------------------------------- #
# Primitive 2: integer-bearing phrase before an anchor                         #
# --------------------------------------------------------------------------- #
def extract_phrase_with_integer(
    page_text: str,
    anchor: str,
    lead_words: tuple[str, ...] = ("approximately", "about", "roughly"),
) -> str | None:
    """
    Return a verbatim phrase of the form
        "<lead_word> <integer-with-commas> ... <anchor>"
    where `<lead_word>` is one of `lead_words` and `<anchor>` is the fixed
    tail (e.g. "full-time and part-time employees").

    Works generically: "approximately 1,576,000 full-time and part-time
    employees", "approximately 1,500,000 full-time and part-time employees",
    "about 1,608,000 full-time and part-time employees" — all match without
    any literal count being baked in.
    """
    t = _norm(page_text)
    if anchor not in t:
        return None

    lead_alt = "|".join(re.escape(w) for w in lead_words)
    # (?:\S+\s+){0,6} — at most 6 tokens of filler between the lead word and
    # the anchor (covers phrases like "approximately 1,576,000 full-time and
    # part-time employees" where the digit itself is the only filler, and
    # short variations like "approximately 1.5 million full-time ...").
    pattern = re.compile(
        rf"(?:{lead_alt})\s+[\d.,]+(?:\s+\S+){{0,4}}\s+{re.escape(anchor)}",
        re.IGNORECASE,
    )
    m = pattern.search(t)
    return m.group(0) if m else None


# --------------------------------------------------------------------------- #
# Primitive 3: value on a "Label  Value" row                                   #
# --------------------------------------------------------------------------- #
def extract_row_value_by_label(
    page_text: str,
    row_label: str,
    value_pattern: str,
) -> str | None:
    """
    On a "two-column" line like

        Other equipment                           Three to ten years

    return the value cell verbatim. `value_pattern` is a regex describing the
    *shape* of the value (e.g. "X to Y years"), not the value itself.

    The function requires `row_label` and a regex match for `value_pattern`
    on the same line; pdfplumber renders such rows as a single line with
    run-of-spaces between cells.
    """
    value_re = re.compile(value_pattern)
    for line in page_text.splitlines():
        if row_label not in line:
            continue
        # Skip the label and search the remainder of the line so a label that
        # happens to contain digits doesn't confuse the value regex.
        tail = line.split(row_label, 1)[1]
        m = value_re.search(tail)
        if m:
            return m.group(0).strip()
    return None
