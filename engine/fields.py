"""Schema-as-code: the target fields for a 10-K-shaped filing.

This module defines *what* to extract and *how to locate it by structure* —
never by literal value. That means the same `FieldSpec` list works on
Amazon's FY2024 10-K, FY2025 10-K, and any future filing whose layout
follows the same template (Item/Note headings, segment table, useful-life
table, Human-Capital paragraph, Item 7A FX paragraph).

Design rules enforced here:

  1. No dollar amounts, employee counts, or year-ranges are hard-coded in
     `predicate` or `verbatim_rule`. Every value comes out of the document
     being parsed.
  2. Predicates use *structural markers*: item titles, note titles, row
     labels, anchor phrases — the kind of thing that survives a re-filing.
  3. Every candidate string still runs through the verbatim gate in
     `engine.verify`, so even a buggy regex cannot fabricate an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from engine.patterns import (
    extract_row_rightmost_money,
    extract_phrase_with_integer,
    extract_row_value_by_label,
)


@dataclass
class FieldSpec:
    key: str
    kind: Literal["numeric", "text"]
    question: str
    anchor_terms: list[str]
    # Deterministic backend: returns True if the page contains the answer.
    predicate: Callable[[str], bool]
    # How to extract the verbatim string once the page is located.
    verbatim_rule: Callable[[str], str | None]
    # Citation metadata overrides (for fields where the table name is canonical).
    item_title: str | None = None
    note_title: str | None = None
    section_title: str | None = None
    table_name: str | None = None
    row: str | None = None
    column: str | None = None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# Verbatim rules — all structural, none value-coupled                          #
# --------------------------------------------------------------------------- #
def _verbatim_aws_net_sales(text: str) -> str | None:
    """The AWS 'Net sales' row in the segment table — rightmost (most recent)
    dollar column. No literal amount required."""
    return extract_row_rightmost_money(
        text, row_label="Net sales", must_follow="AWS"
    )


def _verbatim_headcount(text: str) -> str | None:
    """'approximately <N> full-time and part-time employees' — extracted by
    shape. Whatever integer the filing prints is what comes out; no literal
    count is baked into the rule."""
    return extract_phrase_with_integer(
        text, anchor="full-time and part-time employees"
    )


def _verbatim_other_equipment(text: str) -> str | None:
    """The 'Other equipment' row of the useful-life table. The value pattern
    describes the SHAPE of the cell ('<Word> to <word> years') — the actual
    lifespan is whatever the table prints on the row. Nothing numeric is
    baked in."""
    return extract_row_value_by_label(
        text,
        row_label="Other equipment",
        value_pattern=r"[A-Z][a-z]+(?:\s+to\s+[a-z]+)?\s+years?",
    )


def _verbatim_retail_competition(text: str) -> str | None:
    from engine.verify import extract_sentence
    return extract_sentence(
        text, "principal competitive factors in our retail businesses"
    )


def _verbatim_fx(text: str) -> str | None:
    from engine.verify import extract_sentence
    return extract_sentence(
        text,
        "internationally-focused stores and AWS are exposed to foreign exchange",
    )


# --------------------------------------------------------------------------- #
# Structural predicates — locate the RIGHT page without knowing the ANSWER     #
# --------------------------------------------------------------------------- #
def _pred_aws_net_sales(t: str) -> bool:
    # Must be the AWS *data* page, not the Note-10 intro page. We require:
    #   * "AWS" appears as a segment header,
    #   * a "Net sales" row,
    #   * at least one dollar amount on that row (so intro pages that
    #     mention AWS prose but don't yet show the table are skipped).
    if "AWS" not in t or "Net sales" not in t:
        return False
    return extract_row_rightmost_money(t, "Net sales", must_follow="AWS") is not None


def _pred_headcount(t: str) -> bool:
    return extract_phrase_with_integer(t, "full-time and part-time employees") is not None


def _pred_other_equipment(t: str) -> bool:
    # The page that narrates what "Other equipment" contains AND has the
    # useful-life row. The narrative sentence is identical across years,
    # but the row value changes year to year.
    if "Other equipment consists primarily of" not in t:
        return False
    return _verbatim_other_equipment(t) is not None


def _pred_retail(t: str) -> bool:
    return "principal competitive factors in our retail businesses" in _norm(t)


def _pred_fx(t: str) -> bool:
    return (
        "Foreign Exchange Risk" in t
        and "internationally-focused stores and AWS are exposed" in _norm(t)
    )


# --------------------------------------------------------------------------- #
# Field list                                                                   #
# --------------------------------------------------------------------------- #
AMAZON_10K_FIELDS: list[FieldSpec] = [
    FieldSpec(
        key="aws_net_sales",
        kind="numeric",
        question="Net Sales of the AWS segment for the most recent fiscal year (USD millions).",
        anchor_terms=["AWS", "Net sales", "Segment Information"],
        predicate=_pred_aws_net_sales,
        verbatim_rule=_verbatim_aws_net_sales,
        item_title="Item 8 — Financial Statements and Supplementary Data",
        note_title="Note 10 — SEGMENT INFORMATION",
        table_name="Information on reportable segments and reconciliation to consolidated net income (in millions)",
        row="AWS — Net sales",
        column="Most recent fiscal year (rightmost column in the source table)",
    ),
    FieldSpec(
        key="employee_headcount",
        kind="numeric",
        question="Total full-time and part-time employees at fiscal year end.",
        anchor_terms=["full-time and part-time employees", "Human Capital"],
        predicate=_pred_headcount,
        verbatim_rule=_verbatim_headcount,
        item_title="Item 1 — Business",
        section_title="Human Capital",
    ),
    FieldSpec(
        key="property_equipment_useful_life_fulfillment_equipment",
        kind="text",
        question=(
            "Estimated useful life for Fulfillment equipment. Amazon classifies "
            "fulfillment equipment under the 'Other equipment' row of the "
            "useful-life table — extract that row's value verbatim."
        ),
        anchor_terms=["Fulfillment equipment", "Other equipment", "Estimated useful life"],
        predicate=_pred_other_equipment,
        verbatim_rule=_verbatim_other_equipment,
        item_title="Item 8 — Financial Statements and Supplementary Data",
        note_title="Note 1 — DESCRIPTION OF BUSINESS, ACCOUNTING POLICIES, AND SUPPLEMENTAL DISCLOSURES",
        table_name="Property and equipment — Estimated useful life",
        row="Other equipment",
        column="Estimated useful life",
    ),
    FieldSpec(
        key="primary_retail_competition",
        kind="text",
        question="Exact sentence describing principal competitive factors in retail.",
        anchor_terms=["principal competitive factors", "retail businesses"],
        predicate=_pred_retail,
        verbatim_rule=_verbatim_retail_competition,
        item_title="Item 1 — Business",
        section_title="Competition",
    ),
    FieldSpec(
        key="foreign_exchange_risk",
        kind="text",
        question="Exact sentence summarizing primary FX risk in Item 7A.",
        anchor_terms=["Foreign Exchange Risk", "exposed to foreign exchange rate"],
        predicate=_pred_fx,
        verbatim_rule=_verbatim_fx,
        item_title="Item 7A — Quantitative and Qualitative Disclosures About Market Risk",
        section_title="Foreign Exchange Risk",
    ),
]
