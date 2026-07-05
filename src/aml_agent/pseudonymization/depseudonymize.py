"""
De-pseudonymization for human review.

Mirrors service.py in the reverse direction. Used exclusively on the
outbound path from agent → human reviewer: once the LLM produces a case
file / narrative, we resolve every token back to its real value so a
compliance officer sees actual customer names and account refs, not
opaque tokens.

Two entry points:
    resolve_tokens_in_text(db, s)     -> free-form narrative
    depseudonymize_case_file(db, obj) -> structured dict / list

Deliberate scope: this module is NOT called anywhere in the LLM-facing
pipeline. Its callers are the human review queue and audit export tools.
Enforcing that boundary in code review is how PII stays out of the LLM.
"""

import re
from typing import Any

from sqlalchemy.orm import Session

from aml_agent.pseudonymization.tokenizer import PREFIXES, depseudonymize


# Reverse map: prefix -> entity_type. Built from the tokenizer's PREFIXES
# so adding a new type in one place propagates here automatically.
_PREFIX_TO_TYPE: dict[str, str] = {v: k for k, v in PREFIXES.items()}


# Token pattern: <PREFIX>_<hex>, prefixes bounded by known set to avoid
# false positives on unrelated ALL_CAPS_STRING_LIKE_THINGS.
# Hex length: 8 chars (matches _TOKEN_BYTES=4 in tokenizer.py). Kept
# as a range to tolerate future increases without re-releasing this file.
_PREFIX_ALTERNATION = "|".join(re.escape(p) for p in _PREFIX_TO_TYPE)
_TOKEN_RE = re.compile(rf"\b({_PREFIX_ALTERNATION})_([0-9a-f]{{6,16}})\b")


def resolve_tokens_in_text(db: Session, text: str) -> str:
    """
    Replace every recognised token in `text` with its real value.

    Unknown tokens are left in place — safer than replacing with a
    placeholder that could be confused with real content. If the LLM
    hallucinated a token that doesn't exist in the map, human review
    sees the unresolved token verbatim and treats it accordingly.
    """
    if not text:
        return text

    def _sub(m: re.Match) -> str:
        prefix, _hex = m.group(1), m.group(2)
        entity_type = _PREFIX_TO_TYPE[prefix]
        real = depseudonymize(db, entity_type, m.group(0))
        return real if real is not None else m.group(0)

    return _TOKEN_RE.sub(_sub, text)


def depseudonymize_case_file(db: Session, obj: Any) -> Any:
    """
    Walk a nested dict / list / string structure and resolve tokens.

    Recurses through dicts and lists, leaves other primitives alone.
    Strings pass through resolve_tokens_in_text — the same function
    handles both a raw string field and a narrative embedded in a
    larger structure, so token detection stays consistent.

    Returns a new object; input is not mutated. Preserves original keys
    and non-string values as-is.
    """
    if isinstance(obj, dict):
        return {k: depseudonymize_case_file(db, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [depseudonymize_case_file(db, item) for item in obj]
    if isinstance(obj, str):
        return resolve_tokens_in_text(db, obj)
    return obj