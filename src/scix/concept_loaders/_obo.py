"""Minimal streaming OBO 1.2 / 1.4 parser used by ChEBI and GO loaders.

Reads an OBO flat file one ``[Term]`` stanza at a time. We do not need a
full ontology graph; the loader only consumes a few tags per term:

    id, name, def, synonym, alt_id, is_a, is_obsolete, namespace, xref

Nothing is added to project dependencies — the OBO format is line-based
and tractable in pure Python. References:

  - https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html
  - https://www.geneontology.org/docs/download-ontology/
"""

from __future__ import annotations

import gzip
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Synonym scope tag. OBO 1.2 syntax:
#   synonym: "term" SCOPE [TYPE] [xrefs]
# We capture the quoted text and the scope; ignore type and xrefs.
_SYNONYM_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s+([A-Z]+)\b')

# is_a tag: ``is_a: ID ! human readable name``. We keep the ID only.
_IS_A_RE = re.compile(r"^([^\s!]+)")

# def tag: ``def: "definition text" [xref1, xref2, ...]``.
_DEF_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class OboTerm:
    """A parsed ``[Term]`` stanza, restricted to the tags we use."""

    id: str
    name: str
    namespace: str | None = None
    definition: str | None = None
    synonyms: tuple[str, ...] = ()
    alt_ids: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    xrefs: tuple[str, ...] = ()
    is_obsolete: bool = False
    extra: dict[str, list[str]] = field(default_factory=dict)


def _unescape_quoted(s: str) -> str:
    """Reverse OBO's backslash escaping inside quoted strings."""
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _strip_xref_trailer(value: str) -> str:
    """Drop ``! comment`` and ``{trailing-modifier}`` trailers from an xref."""
    s = value.split("!", 1)[0]
    brace = s.find("{")
    if brace != -1:
        s = s[:brace]
    return s.strip()


def _open(path: Path):
    """Open .obo or .obo.gz transparently."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_terms(path: Path) -> Iterator[OboTerm]:
    """Yield every ``[Term]`` stanza in ``path`` (skipping ``[Typedef]``).

    Obsolete terms are still yielded; loaders decide whether to keep them.
    """
    with _open(path) as fh:
        stanza_kind: str | None = None
        buf: dict[str, list[str]] = {}
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                if stanza_kind == "Term" and buf.get("id"):
                    yield _build_term(buf)
                stanza_kind = line[1:-1]
                buf = {}
                continue
            # Header lines (no stanza yet) are ignored.
            if stanza_kind != "Term":
                continue
            if ":" not in line:
                continue
            tag, _, value = line.partition(":")
            tag = tag.strip()
            value = value.strip()
            # Strip inline ``! comment`` trailers on simple-value tags
            # (id, is_a, alt_id). Multi-segment tags like def/synonym
            # parse the trailer themselves.
            if tag in ("id", "is_a", "alt_id") and "!" in value:
                value = value.split("!", 1)[0].strip()
            buf.setdefault(tag, []).append(value)

        if stanza_kind == "Term" and buf.get("id"):
            yield _build_term(buf)


def _build_term(buf: dict[str, list[str]]) -> OboTerm:
    raw_id = buf.get("id", [""])[0]
    name = (buf.get("name") or [""])[0]

    namespace = (buf.get("namespace") or [None])[0]

    definition: str | None = None
    if "def" in buf:
        m = _DEF_RE.match(buf["def"][0])
        if m:
            definition = _unescape_quoted(m.group(1))

    synonyms: list[str] = []
    for syn_line in buf.get("synonym", ()):
        m = _SYNONYM_RE.match(syn_line)
        if not m:
            continue
        text, scope = m.group(1), m.group(2)
        if scope in ("EXACT", "RELATED", "BROAD", "NARROW"):
            synonyms.append(_unescape_quoted(text))

    alt_ids = tuple(v.strip() for v in buf.get("alt_id", ()) if v.strip())

    parents: list[str] = []
    for is_a in buf.get("is_a", ()):
        m = _IS_A_RE.match(is_a)
        if m:
            parents.append(m.group(1))

    xrefs = tuple(_strip_xref_trailer(v) for v in buf.get("xref", ()) if _strip_xref_trailer(v))

    is_obsolete = bool(buf.get("is_obsolete") and buf["is_obsolete"][0].lower() == "true")

    extra = {
        k: list(v)
        for k, v in buf.items()
        if k
        not in {
            "id",
            "name",
            "namespace",
            "def",
            "synonym",
            "alt_id",
            "is_a",
            "xref",
            "is_obsolete",
        }
    }

    return OboTerm(
        id=raw_id,
        name=name,
        namespace=namespace,
        definition=definition,
        synonyms=tuple(synonyms),
        alt_ids=alt_ids,
        parents=tuple(parents),
        xrefs=xrefs,
        is_obsolete=is_obsolete,
        extra=extra,
    )
