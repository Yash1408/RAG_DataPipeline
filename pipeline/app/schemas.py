"""
Pydantic schemas for request/response contracts.
All extraction outputs enforce the `citation` block required by the brief.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class NumericCitation(BaseModel):
    page: int
    item: str | None = None
    note: str | None = None
    table_name: str = Field(..., description="Specific name of the table on the page")
    row: str | None = None
    column: str | None = None
    surrounding_text: str = Field(..., description="Exact text snippet surrounding the answer")


class TextCitation(BaseModel):
    page: int
    item: str | None = None
    section: str | None = None
    surrounding_text: str = Field(..., description="Exact text snippet surrounding the answer")


class NumericExtraction(BaseModel):
    value: float | int
    unit: str | None = None
    verbatim: str = Field(..., description="Exact string as it appears in the PDF")
    citation: NumericCitation


class TextExtraction(BaseModel):
    value_verbatim: str = Field(..., description="Exact text as it appears in the PDF")
    citation: TextCitation


class ExtractionRequest(BaseModel):
    schema_id: str = "amazon_10k_v1"


class ExtractionResponse(BaseModel):
    job_id: str
    tenant_id: str
    status: Literal["completed"]
    document: dict
    extractions: dict
    provenance: dict


class JobStatus(BaseModel):
    job_id: str
    status: str
