"""Section-level embedding pipeline.

This package contains the pipeline that reads ``papers_fulltext.sections`` and
writes per-section embeddings to ``section_embeddings``. Lives apart from
``scix.embed`` (paper-level SPECTER2 / INDUS path) because the lifecycles, the
storage tables, and the model registries are different.

Public entry points are exposed by ``scix.embeddings.section_pipeline``; this
``__init__`` is intentionally minimal so ``python -m scix.embeddings.section_pipeline``
does not double-import the submodule.
"""
