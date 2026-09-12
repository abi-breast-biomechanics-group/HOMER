"""Render the reST roles in HOMER's docstrings as mkdocstrings cross-references.

The docstrings are reST (``docstring_style: sphinx``), but mkdocstrings renders
docstring *bodies* as Markdown, so a role such as
``:class:`~HOMER.mesh.mesh.Mesh``` would otherwise reach the page verbatim.

A role becomes a Markdown cross-reference when its target is something the API
reference actually renders -- a public class or function inside ``HOMER`` -- and
inline code otherwise, so that privates, bare names and third-party types read
correctly instead of producing dead links.  Sphinx's ``~`` prefix (show only the
last component) and its ``text <target>`` form are both honoured.
"""
from __future__ import annotations

import re

from griffe import Extension, Object

ROLE = re.compile(r":(?:py:)?(?:class|meth|func|mod|attr|obj|data|exc):`(?P<body>[^`]+)`")
EXPLICIT = re.compile(r"(?P<text>[^<]+?)\s*<(?P<target>[^>]+)>")


def split(body: str) -> tuple[str, str]:
    """The display text and link target of one role body."""
    explicit = EXPLICIT.fullmatch(body)
    if explicit:
        return explicit.group("text"), explicit.group("target").lstrip("~")
    target = body.lstrip("~")
    return (target.rsplit(".", 1)[-1] if body.startswith("~") else target), target


class RestRoles(Extension):
    """Rewrite reST roles into cross-references on every documented object."""

    def on_object(self, *, obj: Object, loader=None, **kwargs) -> None:
        if obj.docstring is None:
            return
        obj.docstring.value = ROLE.sub(
            lambda m: self._render(m, loader), obj.docstring.value
        )

    def _render(self, match: re.Match, loader) -> str:
        text, target = split(match.group("body").strip())
        return f"[`{text}`][{target}]" if self._renders(target, loader) else f"`{text}`"

    @staticmethod
    def _renders(target: str, loader) -> bool:
        """Whether the API reference carries an anchor for *target*."""
        parts = target.split(".")
        if len(parts) < 2 or parts[0] != "HOMER" or any(p.startswith("_") for p in parts):
            return False
        if loader is None:
            return False
        try:
            resolved = loader.modules_collection[target]
        except (KeyError, AttributeError, TypeError):
            return False
        if resolved.is_alias:
            return False
        return resolved.is_class or resolved.is_function
