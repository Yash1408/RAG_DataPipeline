"""
Local extraction engine.

A single import surface — `from engine import extract_from_pdf` — used by
both the Streamlit demo (streamlit_app.py) and the production FastAPI
workers (pipeline/app/workers/extractor.py). Keeps one canonical pipeline.
"""
from engine.extractor import extract_from_pdf, ExtractionResult

__all__ = ["extract_from_pdf", "ExtractionResult"]
