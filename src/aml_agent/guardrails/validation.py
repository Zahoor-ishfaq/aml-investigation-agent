"""
Input/output validation guardrails.

Two pattern-based checks over the agent's narrative field:

  1. PII-leak detection — a name or long numeric identifier appearing
     verbatim in the narrative means the agent bypassed pseudonymization.
  2. LLM-artifact detection — disclaimer phrases and meta-commentary
     leaking from the model's chat register into the case file.

These are heuristics, not proofs. Regex-based PII detection has known
false-positive and false-negative rates; in production this would be
paired with a dedicated PII tool (e.g. Microsoft Presidio). Both
guardrails return BLOCK on match rather than REQUIRES_REVIEW: a leaked
name is a quality bug in the output, not a borderline judgment call.
"""

import re

from aml_agent.guardrails.base import (
    AgentDecision,
    Guardrail,
    GuardrailDecision,
    GuardrailResult,
    register,
)


# ---------------------------------------------------------------------------
# Token pattern — must match the tokenizer's PREFIXES and _TOKEN_BYTES.
# Kept as its own regex so it can be exempted from the PII scan (valid
# tokens are exactly what we want the narrative to contain).
# ---------------------------------------------------------------------------
_VALID_TOKEN_RE = re.compile(r"\b(CUST|CEXT|AEXT|TEXT)_[0-9a-f]{6,16}\b")

# Bare integers of length >= 5 that aren't part of a valid token or
# a plausible date/amount. Long digit runs in a narrative are a red
# flag for an account number leaking through.
_LONG_DIGIT_RE = re.compile(r"\b\d{5,}\b")

# ProperCase Name Name pattern — two or more capitalised words in a row.
# Genuine false positives include place names ("New York") and org names
# ("United Nations"), but those are rare in an AML case narrative and
# a BLOCK triggering human review on those cases is cheap compared to
# a customer name leaking to logs.
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Word-list of proper nouns exempted from the ProperCase check because
# they show up in legitimate AML narratives without being PII.
_PROPER_NAME_ALLOWLIST = {
    "Financial Action Task", "Financial Action Task Force",
    "New York", "United States", "United Kingdom", "United Nations",
    "Saudi Arabia", "European Union", "Middle East",
    "Fan In", "Fan Out",  # our own typology names
}


@register
class PiiLeakGuardrail(Guardrail):
    """
    Block narratives containing long digit sequences or proper-case
    multi-word phrases that aren't valid tokens or allowlisted terms.
    """
    code = "PII_LEAK_SUSPECTED"
    description = "Blocks narratives with unmasked digit sequences or proper-case names"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if not decision.narrative:
            # Missing narrative is handled by NarrativeRequiredGuardrail;
            # this one only cares about narratives with content.
            return GuardrailResult(self.code, GuardrailDecision.PASS)

        text = decision.narrative

        # Strip valid tokens before checking — they're the tokens we WANT
        # in the narrative. Substituting them out avoids false positives
        # on their trailing hex digits.
        stripped = _VALID_TOKEN_RE.sub("", text)

        digit_hits = _LONG_DIGIT_RE.findall(stripped)
        if digit_hits:
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                f"narrative contains unmasked digit sequence(s): {digit_hits[:3]}",
            )

        name_hits = [
            m for m in _PROPER_NAME_RE.findall(stripped)
            if m not in _PROPER_NAME_ALLOWLIST
        ]
        if name_hits:
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                f"narrative contains suspected proper-name PII: {name_hits[:3]}",
            )

        return GuardrailResult(self.code, GuardrailDecision.PASS)


# ---------------------------------------------------------------------------
# LLM artifact phrases. Deliberately conservative — only phrases that
# would clearly be out of place in a professional AML case narrative.
# A regulator reading "As an AI, I cannot..." in a SAR justification
# would rightly question the entire pipeline.
# ---------------------------------------------------------------------------
_LLM_ARTIFACT_PATTERNS = [
    re.compile(r"\bas an (AI|assistant|LLM|language model)\b", re.IGNORECASE),
    re.compile(r"\bI (cannot|can't|am unable to|apologize|am sorry)\b", re.IGNORECASE),
    re.compile(r"\bI'?m (an AI|sorry|unable)\b", re.IGNORECASE),
    re.compile(r"\bplease note\b.*\b(I|my|this)\b", re.IGNORECASE),
    re.compile(r"\bas a (large language model|chatbot)\b", re.IGNORECASE),
]


@register
class LlmArtifactGuardrail(Guardrail):
    """
    Block narratives containing LLM-chat-register artifacts.

    These phrases indicate the model slipped out of "AML analyst" role
    into its default "helpful assistant" register — case files must be
    written as an analyst would write them, not as a chatbot's meta-reply.
    """
    code = "LLM_ARTIFACT_DETECTED"
    description = "Blocks narratives containing LLM disclaimer / meta-commentary phrases"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if not decision.narrative:
            return GuardrailResult(self.code, GuardrailDecision.PASS)

        for pattern in _LLM_ARTIFACT_PATTERNS:
            match = pattern.search(decision.narrative)
            if match:
                return GuardrailResult(
                    self.code, GuardrailDecision.BLOCK,
                    f"narrative contains LLM artifact phrase: {match.group(0)!r}",
                )

        return GuardrailResult(self.code, GuardrailDecision.PASS)