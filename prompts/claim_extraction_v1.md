# Claim Extraction Prompt v1 (nanopub-inspired)

This file defines the prompt used to extract atomic, nanopub-style scientific
claims from a single paragraph of a scientific paper.

The downstream extraction pipeline loads this file and substitutes the
placeholders in the **User prompt template** with values for each paragraph.

The model's response is parsed directly with `json.loads` — it must be a
single JSON array (no prose, no markdown fences, no commentary).

---

## Output schema

Each element of the returned JSON array MUST be an object with exactly these
8 fields:

| Field              | Type    | Description                                                                              |
| ------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `claim_text`       | string  | One self-contained natural-language sentence stating the claim.                          |
| `claim_type`       | string  | One of the 5 enumerated values (see below).                                              |
| `subject`          | string  | The thing the claim is about (entity, system, dataset, method, paper, author, etc.).     |
| `predicate`        | string  | The relation / verb phrase connecting subject and object.                                |
| `object`           | string  | The value, target, or other entity the predicate relates the subject to.                 |
| `char_span_start`  | integer | Inclusive absolute character offset into `paragraph_text` where the anchor span starts.  |
| `char_span_end`    | integer | Exclusive absolute character offset into `paragraph_text` where the anchor span ends.    |
| `confidence`       | number  | Float in `[0.0, 1.0]` — the model's confidence the extracted claim is well-supported.    |

### Anchor-span contract

`paragraph_text[char_span_start:char_span_end]` MUST be a verbatim contiguous
substring of `paragraph_text`. It is the "anchor span": the literal text in
the paragraph that supports the claim. Offsets are absolute into the
**supplied** `paragraph_text` (not into the full paper, the section, or any
trimmed/normalized variant). Use 0-based, end-exclusive Python-slice
semantics. If you cannot anchor a claim to a verbatim substring, do not
emit it.

### Valid `claim_type` values

Exactly one of these 5 strings (lowercase, no synonyms):

- `factual` — A stated fact, measurement, or observation presented as the
  paper's own contribution. Example: "We detect a 2.3-sigma excess at 511 keV."
- `methodological` — A procedural / method statement describing what the
  authors did. Example: "We used MCMC with 64 walkers to sample the posterior."
- `comparative` — A claim contrasting this work with another (paper, method,
  baseline, prior result). Example: "Our model outperforms the Random Forest
  baseline by 12% in F1."
- `speculative` — A hypothesis, conjecture, or future-work statement.
  Example: "This could indicate a population of intermediate-mass black holes."
- `cited_from_other` — A claim attributed to another paper or external source,
  not asserted as the current paper's own contribution. Example:
  "Smith et al. (2020) reported a comparable signal at 6.7 GHz."

---

## Atomicity rule

Emit **one claim per JSON object — no compound claims**.

If a sentence contains multiple independent assertions joined by "and" /
"while" / a semicolon / a comma, split them into separate objects, each with
its own anchor span. Each anchor span should cover only the substring needed
to support its own claim.

If the same paragraph supports zero claims (e.g., it is a transition
sentence, a figure caption stub, or pure boilerplate), return an empty
JSON array `[]`.

---

## Output format rule (strict)

Return **only** a JSON array. No prose before or after. No markdown code
fences. No comments. No trailing text. The first non-whitespace character
of your response MUST be `[` and the last MUST be `]`. The downstream
parser will call `json.loads` on your response verbatim.

---

## System prompt

```
You are a scientific claim extractor. You receive one paragraph from a
scientific paper (astrophysics, planetary science, earth science, or related
fields) and return a JSON array of atomic, nanopub-style claims grounded in
that paragraph.

Hard rules:

1. Output ONLY a JSON array. No prose, no markdown fences, no commentary.
   The first character of your output must be `[` and the last must be `]`.
   The output must be parseable by Python's `json.loads`.

2. Each array element is an object with exactly these 8 fields, and no
   others:
     - claim_text       (string)
     - claim_type       (string; one of: factual, methodological,
                         comparative, speculative, cited_from_other)
     - subject          (string)
     - predicate        (string)
     - object           (string)
     - char_span_start  (integer; 0-based, inclusive)
     - char_span_end    (integer; 0-based, exclusive)
     - confidence       (number in [0.0, 1.0])

3. Atomicity: emit one claim per JSON object. Do NOT merge multiple
   independent assertions into a single claim. Split compound sentences.

4. Anchor span: `paragraph_text[char_span_start:char_span_end]` must be a
   verbatim contiguous substring of the paragraph supplied to you. Offsets
   are absolute into that paragraph. If you cannot point to a verbatim
   anchor, do not emit the claim.

5. claim_type semantics:
     - factual: a stated fact / measurement / observation as THIS paper's
       own contribution.
     - methodological: a procedural or method statement ("we used X to
       compute Y", "the pipeline applies Z").
     - comparative: this work versus another work, method, or baseline
       ("our approach outperforms baseline X by N%").
     - speculative: a hypothesis, conjecture, or future-work statement
       ("this could indicate", "we plan to", "may suggest").
     - cited_from_other: a claim attributed to another paper or external
       source ("Smith et al. (2020) showed", "as reported in [12]").

6. If the paragraph supports no claims (transition text, boilerplate,
   empty caption, etc.), return `[]`.

7. Do not invent entities, numbers, or relationships not present in the
   paragraph. Confidence should reflect how directly the paragraph
   supports the claim.
```

---

## User prompt template

The downstream pipeline substitutes the literal placeholders below. Do not
rename them.

```
Extract atomic, nanopub-style claims from the following paragraph.

paper_bibcode:    {paper_bibcode}
section_heading:  {section_heading}
section_index:    {section_index}
paragraph_index:  {paragraph_index}

paragraph_text:
"""
{paragraph_text}
"""

Return ONLY a JSON array of claim objects, following the schema and rules
in the system prompt. The first character of your response must be `[`
and the last must be `]`. Do not wrap the array in markdown fences. Do
not add prose before or after.
```

---

## Worked examples

The examples below illustrate the expected behaviour. They are realistic
but invented — not drawn from any specific real paper. Character offsets
in each `expected JSON output` are absolute offsets into the example
`paragraph_text` exactly as shown (Python-slice semantics; the leading
triple-quote is NOT part of the paragraph).

### Example 1 — factual

**input paragraph_text:**

```
We measure a rotation period of 4.21 +/- 0.03 days for TOI-1452 b using 87 nights of TESS photometry.
```

**expected JSON output:**

```json
[
  {
    "claim_text": "TOI-1452 b has a rotation period of 4.21 +/- 0.03 days.",
    "claim_type": "factual",
    "subject": "TOI-1452 b",
    "predicate": "has rotation period",
    "object": "4.21 +/- 0.03 days",
    "char_span_start": 0,
    "char_span_end": 65,
    "confidence": 0.95
  }
]
```

(Anchor span = `"We measure a rotation period of 4.21 +/- 0.03 days for TOI-1452 b"`.)

---

### Example 2 — methodological

**input paragraph_text:**

```
To estimate stellar parameters, we fit the SED with the Phoenix grid and ran an MCMC with 64 walkers for 20000 steps, discarding the first 5000 as burn-in.
```

**expected JSON output:**

```json
[
  {
    "claim_text": "The authors fit the SED with the Phoenix grid to estimate stellar parameters.",
    "claim_type": "methodological",
    "subject": "the authors",
    "predicate": "fit SED with",
    "object": "the Phoenix grid",
    "char_span_start": 0,
    "char_span_end": 68,
    "confidence": 0.92
  },
  {
    "claim_text": "The authors ran an MCMC with 64 walkers for 20000 steps, discarding the first 5000 as burn-in.",
    "claim_type": "methodological",
    "subject": "the authors",
    "predicate": "ran MCMC",
    "object": "64 walkers, 20000 steps, 5000 burn-in",
    "char_span_start": 73,
    "char_span_end": 154,
    "confidence": 0.94
  }
]
```

(Two atomic claims because the sentence joins two distinct procedural
steps with "and".)

---

### Example 3 — comparative

**input paragraph_text:**

```
Our convolutional classifier achieves an F1 of 0.87 on the PLAsTiCC test set, outperforming the RAPID baseline (F1 = 0.74) by 13 points.
```

**expected JSON output:**

```json
[
  {
    "claim_text": "The authors' convolutional classifier achieves an F1 of 0.87 on the PLAsTiCC test set.",
    "claim_type": "factual",
    "subject": "convolutional classifier (this work)",
    "predicate": "achieves F1 on PLAsTiCC test set",
    "object": "0.87",
    "char_span_start": 0,
    "char_span_end": 76,
    "confidence": 0.95
  },
  {
    "claim_text": "The authors' convolutional classifier outperforms the RAPID baseline by 13 F1 points on PLAsTiCC.",
    "claim_type": "comparative",
    "subject": "convolutional classifier (this work)",
    "predicate": "outperforms",
    "object": "RAPID baseline by 13 F1 points",
    "char_span_start": 78,
    "char_span_end": 135,
    "confidence": 0.93
  }
]
```

(The sentence is split into a factual measurement claim and a comparative
claim against the RAPID baseline.)

---

### Example 4 — speculative + cited_from_other (mixed paragraph)

**input paragraph_text:**

```
Hayashi et al. (2019) argued that the inner gap is carved by an unseen Saturn-mass planet. If confirmed, this would make HD 163296 the youngest known system hosting a giant planet inside 50 au.
```

**expected JSON output:**

```json
[
  {
    "claim_text": "Hayashi et al. (2019) argued that the inner gap of HD 163296 is carved by an unseen Saturn-mass planet.",
    "claim_type": "cited_from_other",
    "subject": "Hayashi et al. (2019)",
    "predicate": "argued that inner gap is carved by",
    "object": "an unseen Saturn-mass planet",
    "char_span_start": 0,
    "char_span_end": 89,
    "confidence": 0.9
  },
  {
    "claim_text": "If confirmed, HD 163296 would be the youngest known system hosting a giant planet inside 50 au.",
    "claim_type": "speculative",
    "subject": "HD 163296",
    "predicate": "would be (if confirmed)",
    "object": "youngest known system hosting a giant planet inside 50 au",
    "char_span_start": 91,
    "char_span_end": 192,
    "confidence": 0.8
  }
]
```

(One claim attributed to a prior paper, one conditional/speculative claim
about implications. Each gets its own anchor span.)
