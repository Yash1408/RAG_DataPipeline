# Debrief — The Hardest Technical Challenge in This Document

## 1. What makes a 10-K structurally hostile to RAG

An Amazon 10-K is 125 pages, but it is not 125 pages of prose. It is a
**graph of cross-references** between four distinct structural contexts
that a reader fuses together automatically and that a retrieval system
does not:

| Context | Examples in this document |
|---|---|
| Item- and Note-level headings that span dozens of pages | *Item 8 — Financial Statements and Supplementary Data* opens on p. 64 and owns every page until p. 115. *Note 10 — SEGMENT INFORMATION* starts on p. 105 and its data tables run onto p. 107 and p. 108. |
| Tables with multi-year columns and run-over rows | The segment table on p. 107 has three fiscal-year columns (2023 / 2024 / 2025); the row header "Net sales" appears under each segment block (AWS, North America, International). |
| Prose sentences that *disambiguate* adjacent table rows | On p. 75, the sentence *"Other equipment consists primarily of fulfillment equipment"* is the only thing that tells the reader which row of the useful-life table governs Amazon's fulfillment network. |
| Section names that repeat under different Items | "Foreign Exchange Risk" appears both in *Item 1A — Risk Factors* and in *Item 7A — Quantitative and Qualitative Disclosures About Market Risk* — with different content and a different intended audience. |

A naive chunk-and-embed pipeline flattens all four of these into a single
token stream, and the joins between them disappear. Every one of the five
target fields in this assignment trips on at least one of those joins.
The "Fulfillment equipment useful life" field trips all four
simultaneously and is the clearest vehicle to describe the problem.

## 2. The single hardest field: *Fulfillment equipment useful life*

**The question.** *"What is the estimated useful life for Fulfillment
equipment?"*

**The document reality.** The useful-life table on page 75 has **no row
labeled `Fulfillment equipment`**. Its four rows are *Buildings*, *Servers
and networking equipment*, *Heavy equipment*, and *Other equipment*. The
answer is hidden in the sentence that precedes the table:

> *"Heavy equipment consists primarily of assets that support the
> infrastructure of our fulfillment network and data centers.
> **Other equipment consists primarily of fulfillment equipment.**
> Depreciation … is recorded on a straight-line basis over the
> estimated useful lives of the assets…"*

Only after reading that sentence does the reader know that the *Other
equipment* row — *Three to ten years* — is the one that governs
fulfillment-equipment depreciation. *Heavy equipment* is a deliberate
distractor: its definition also contains the phrase "fulfillment network"
and its row value *Ten to thirteen years* is a plausible-sounding wrong
answer.

### Why the obvious pipelines fail on this single field

**Failure mode 1 — term-to-row mismatch.** Any system that retrieves the
table and then does row-label lookup for "Fulfillment equipment" returns
`null`, or silently picks the closest-looking row.

**Failure mode 2 — chunk boundary destroys the join.** 512-token chunking
typically places the narrative in one chunk, the table in the next, and
the *Note 1* heading in a third. The model never sees them together, so
even a smart model cannot make the inference.

**Failure mode 3 — flatten-to-text destroys column alignment.** Converting
`<table>` to `"Buildings Lesser of forty years... Heavy equipment Ten to
thirteen years Other equipment Three to ten years"` produces a string
where "Other equipment" might bind to the wrong span.

**Failure mode 4 — plausible hallucination.** A temperature > 0 model
under prompt pressure will confidently invent *"Three to five years"*
because the phrase has no literal anchor in the document to fail against.

## 3. How the pipeline overcomes it — layer by layer

The extractor's response to this field is not one trick but six cooperating
constraints. Each of them also happens to protect the other four fields.

**Layer 1 — Structure-preserving layout parser** ([`engine/layout_parser.py`](engine/layout_parser.py)).
pdfplumber returns tables as `TableBlock(header, rows, page_no)` objects,
not as flattened text. Column alignment is retained, so "Other equipment"
unambiguously pairs with the cell in the *Estimated useful life* column
regardless of how many spaces pdfplumber chose to put between them.

**Layer 2 — Page-as-chunk retrieval, not 512-token chunks**
([`engine/local_embeddings.py`](engine/local_embeddings.py)). The entire
useful-life discussion — prose sentence, table header, four rows —
fits on page 75. Retrieving at page granularity means the disambiguating
sentence and the correct table row **always arrive at the LLM in the same
prompt**. Chunk-boundary loss is impossible by construction. Pages are
also the granularity the final citation needs, so there is no
chunk-to-page reconciliation step.

**Layer 3 — Stateful Item/Note tagging during parse**
([`engine/layout_parser.py:47–74`](engine/layout_parser.py)). A small
automaton walks the pages and stamps each one with the last-seen `Item N.`
and `Note N —` heading. This is what lets the *Foreign Exchange Risk*
citation correctly say *Item 7A* instead of *Item 1A* — the page itself
carries its Item context, so the retriever does not have to re-derive it.
The same mechanism lets *AWS Net Sales* be cited against *Note 10* rather
than against a random table on another page that happens to contain the
same row label.

**Layer 4 — Hybrid retrieval with anchor terms across both sides of the
join**. The field spec for this question declares
`anchor_terms = ["Fulfillment equipment", "Other equipment", "Estimated
useful life"]`. BM25 pulls in pages where the literal phrase *Fulfillment
equipment* appears in prose, embeddings pull in pages where the
*concept* of a useful-life table lives, and Reciprocal Rank Fusion picks
the intersection. Page 75 is the only page that scores highly on both
sides.

**Layer 5 — Generic structural primitives, not hard-coded values**
([`engine/patterns.py`](engine/patterns.py)). The deterministic rule for
this field is `extract_row_value_by_label(row_label="Other equipment",
value_pattern=r"[A-Z][a-z]+(?:\s+to\s+[a-z]+)?\s+years?")`. The pattern
describes the *shape* of the cell — *<Word> to <word> years* — not the
literal *Three to ten years*. If a future filing changes the lifespan to
*Five to twelve years*, the extractor finds it with zero code changes.

**Layer 6 — The verbatim gate as the audit floor**
([`engine/verify.py`](engine/verify.py)). Before any answer leaves the
pipeline, `verify_verbatim(value, page_text)` asserts that the value —
whitespace-normalized — exists character-for-character on the cited
page. If the LLM returns *"Three to five years"* under prompt pressure,
the string is not on page 75, the assertion fails, the value is dropped,
and the field is either re-attempted against the next candidate page or
returned as `not_found`. **There is no codepath that can emit a value not
on the cited page.**

## 4. The same structural hazards, applied to the other four fields

| Field | Structural hazard | Pipeline defense |
|---|---|---|
| AWS Net Sales | Appears in **two** tables — the MD&A summary on p. 43 and the Segment Information table on p. 107 — with the **same row label** and the same dollar amount. The brief requires the segment-level table. | Item/Note tagging + the `must_follow="AWS"` argument to `extract_row_rightmost_money` confine the search to lines under the AWS segment heading of *Note 10*, and the cited `note_title` is captured from the parser's stateful tagger, not a hard-coded string. |
| AWS Net Sales column choice | The segment table has columns for FY2023, FY2024, and FY2025. Picking the wrong column is off by ~30 %. | The primitive takes the **rightmost** `$`-prefixed token on the row. 10-K tables are chronological left-to-right with the most-recent year on the right; the rule is year-agnostic and future-proof for FY2026/2027 filings. |
| Employee headcount | Phrased as prose ("approximately 1,576,000 full-time and part-time employees"), not as a table cell. | Regex primitive `extract_phrase_with_integer` matches `approximately <N> full-time and part-time employees` — the count is whatever the document prints; nothing about 1.57 M is baked in. |
| Primary retail competition | A single sentence buried in a paragraph of competition discussion on p. 6. | `extract_sentence(page_text, anchor)` walks backward to the previous `. ` and forward to the next `. `, so the exact sentence is emitted verbatim regardless of how pdfplumber wrapped the surrounding lines. |
| Foreign Exchange Risk | The phrase "Foreign Exchange Risk" appears as a section heading in **both** *Item 1A — Risk Factors* and *Item 7A — Quantitative and Qualitative Disclosures About Market Risk*. | The field-spec predicate co-requires `internationally-focused stores and AWS are exposed` — a phrase that appears only in Item 7A — and the citation's `item_title` comes from the parser's stateful tagger, guaranteeing p. 56 over p. 15. |

## 5. What the architecture does with failures instead of fabricating

An audit-grade system is defined by what it does when it is unsure. This
pipeline does three things and refuses to do a fourth:

1. **It drops the field.** If every candidate page fails the verbatim
   gate, the field is simply omitted from the response. A consumer of the
   JSON sees that the key is missing and can treat it as "not found"
   rather than trusting a confident-sounding hallucination.
2. **It degrades gracefully.** A flaky or absent LLM opens the
   `LLM_BREAKER`; the orchestrator falls through to the deterministic
   rule for the affected field. A job that would have been *5/5 via LLM*
   becomes *3/5 via LLM + 2/5 via deterministic* instead of *0/5, failed*.
3. **It surfaces the failure in the Dashboard** ([`streamlit_app.py`](streamlit_app.py) / [`engine/store.py`](engine/store.py)).
   When a whole document fails (corrupt PDF, OCR timeout, …) the job is
   marked `failed` in the store with the exception type and message, and
   appears as a red row in the dashboard with its reason. No upload is
   ever silently dropped.
4. **It never re-asks the LLM with a hint toward the answer.** There is
   no "try harder" retry codepath that could coerce a fabrication. The
   only retry is against the **next** candidate page, with the same
   prompt.

## 6. One-line summary

> *The hardest problem in this document is not reading any single page —
> it is recognizing that the answer to the user's question lives in the
> **join** between a prose sentence, a table row, and an Item-level
> heading, and refusing to guess when that join is ambiguous. The
> pipeline wins by preserving those joins all the way through parsing,
> retrieval, and extraction, and by gating every answer on a literal
> substring check against its cited page.*
