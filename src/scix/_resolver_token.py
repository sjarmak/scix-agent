"""Module-private sentinel used to gate construction of :class:`EntityLinkSet`.

Only :mod:`scix.resolve_entities` is allowed to import ``_RESOLVER_INTERNAL``
and pass it into :class:`scix.entity_link_set.EntityLinkSet`. Any other caller
that tries to construct an ``EntityLinkSet`` directly will receive a
``TypeError`` at runtime.

This sentinel is NOT re-exported from ``scix.__init__``.
"""

from __future__ import annotations


class _ResolverToken:
    """Opaque marker type for the single-entry-point construction gate."""

    __slots__ = ()


_RESOLVER_INTERNAL: _ResolverToken = _ResolverToken()
