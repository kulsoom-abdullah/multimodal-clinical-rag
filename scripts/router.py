#!/usr/bin/env python3
"""
Shared query-time logic for the Streamlit app and the CLI.

Both entry points previously carried their own copy of the intent regexes and the
answer prompt, and the copies drifted. Everything here is imported by both.

Routing contract
----------------
`decide_route` returns a `RouteDecision` whose `route` is one of:

  STRICT      an identifier was recognised AND exists in the index -> hard metadata filter
  HYBRID      no identifier -> semantic + BM25 ensemble
  UNKNOWN_ID  an NCT ID was named but is not in this corpus -> answer that, do not search

Candidate identifiers are validated against the identifiers actually present in the
index before they are allowed to route. A bare regex cannot distinguish a protocol
number from ordinary hyphenated clinical English ("non-small-cell", "intent-to-treat"),
and an unvalidated match sends those queries into a filter that matches nothing.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

STRICT = "strict"
HYBRID = "hybrid"
UNKNOWN_ID = "unknown_id"

# An NCT ID is unambiguous: 'NCT' plus exactly 8 digits, not glued to more digits.
# Tolerates the space people actually type ("NCT 02423343").
NCT_PATTERN = re.compile(r"\bNCT[\s-]?(\d{8})\b", re.IGNORECASE)

# Protocol numbers are NOT matched by pattern. A regex cannot tell "H9H-MC-JBEF" from
# "non-small-cell", and the shapes in use are too varied to enumerate (205801,
# CA116001, 189-12, KEYNOTE-189, YHCT-HEX-B1, 3000-02-005, 205801/Amendment 08).
# Instead the query is scanned for the protocol numbers actually present in the index.
# This cannot false-positive, and it covers every indexed protocol rather than the
# two that happened to fit a pattern.


@dataclass
class RouteDecision:
    route: str
    filter_key: Optional[str] = None
    filter_value: Optional[str] = None
    reason: str = ""
    #  Candidate that looked like an ID but was not in the index; kept for display.
    rejected_candidate: Optional[str] = None


def _norm(value: str) -> str:
    return value.strip().casefold()


def collect_known_ids(metadatas: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the identifier whitelist from index metadata.

    Protocol identifiers are stored as {value: field}, carrying the field each
    value came from so a match is filtered on the field that actually holds it.
    A value present only in `protocol_number` cannot be filtered as `protocol_id`,
    and doing so routes a recognised identifier into a filter matching nothing.

    Where a value appears under both, `protocol_number` wins: it spans the whole
    protocol, while `protocol_id` names one printed variant of it.
    """
    trials: Set[str] = set()
    protocols: Dict[str, str] = {}
    for meta in metadatas:
        trial = meta.get("trial_id")
        if trial:
            trials.add(str(trial))
        for field in ("protocol_id", "protocol_number"):
            value = meta.get(field)
            if not value or str(value) == "<UNKNOWN>":
                continue
            value = str(value)
            if field == "protocol_number" or value not in protocols:
                protocols[value] = field
    return {"trial_id": trials, "protocol": protocols}


def load_known_ids(vectorstore, batch_size: int = 2000) -> Dict[str, Set[str]]:
    """Page the whole collection once and collect the identifiers present in it."""
    collection = vectorstore._collection
    total = collection.count()
    metadatas = []
    offset = 0
    while offset < total:
        batch = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        metadatas.extend(batch["metadatas"])
        offset += batch_size
    return collect_known_ids(metadatas)


def _match_known(candidate: str, known: Set[str]) -> Optional[str]:
    """Return the indexed spelling of `candidate`, or None if it is not indexed."""
    lookup = {_norm(k): k for k in known}
    return lookup.get(_norm(candidate))


def _find_indexed_protocol(query: str, known_protocols: Dict[str, str]) -> Optional[str]:
    """Find an indexed protocol identifier mentioned in the query.

    Longest first, so '205801/Amendment 08' wins over the bare '205801' it contains.
    Boundaries are checked explicitly because protocol numbers contain '-' and '/',
    which are not word characters to `re`.
    """
    for protocol in sorted(known_protocols, key=len, reverse=True):
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(protocol) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        if pattern.search(query):
            return protocol
    return None


def decide_route(query: str, known_ids: Optional[Dict[str, Set[str]]] = None) -> RouteDecision:
    known_ids = known_ids or {}
    known_trials = known_ids.get("trial_id", set())
    known_protocols = known_ids.get("protocol", {})

    nct_match = NCT_PATTERN.search(query)
    if nct_match:
        candidate = f"NCT{nct_match.group(1)}"
        resolved = _match_known(candidate, known_trials) if known_trials else candidate
        if resolved:
            return RouteDecision(
                STRICT, "trial_id", resolved, f"trial_id {resolved} -> strict filter"
            )
        # A named NCT ID that is not indexed is a question we can answer precisely:
        # broadening to hybrid would answer about some other trial instead.
        return RouteDecision(
            UNKNOWN_ID,
            reason=f"{candidate} is not in this corpus",
            rejected_candidate=candidate,
        )

    protocol = _find_indexed_protocol(query, known_protocols)
    if protocol:
        # Filter on the field this value was found in, not a fixed field name.
        field = known_protocols[protocol]
        return RouteDecision(
            STRICT, field, protocol, f"{field} {protocol} -> strict filter"
        )

    return RouteDecision(HYBRID, reason="no indexed identifier -> hybrid search")


def detect_intent(
    query: str, known_ids: Optional[Dict[str, Set[str]]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Backwards-compatible shim: (intent_type, value), or (None, None)."""
    decision = decide_route(query, known_ids)
    if decision.route == STRICT:
        return decision.filter_key, decision.filter_value
    return None, None


# Shared answer prompt. Guideline 5 is the abstention clause: without it a query that
# retrieves nothing still produces a fluent, cited-looking answer from model priors.
ANSWER_PROMPT = """You are a Senior Clinical Research Associate (CRA) assisting with protocol verification.
Your task is to answer the user's question based *strictly* on the provided context.

GUIDELINES:
1. **Evidence-Based:** Answer only using the provided chunks. Do not use outside knowledge.
2. **Hierarchy of Data:** If you see multiple versions (e.g., Protocol v1.0 vs Amendment 2), prioritize the LATEST information.
3. **Safety First:** If the user asks about Dosing, Exclusion Criteria, or Toxicity Management, quote the text/values exactly.
4. **Visuals:** If the context includes an image description (e.g., "Figure 1", "Flowchart"), refer to it in your answer.
5. **Uncertainty:** If the context does not contain the answer, say so plainly and do not fill the gap from prior knowledge. Never invent a citation.
6. **Citations:** End your answer with the specific Source Documents used (e.g., "Source: SAP_001.pdf").

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (Structured for a Clinical Audience):"""

# Shown instead of calling the model when retrieval returns nothing.
NO_CONTEXT_MESSAGE = (
    "No documents matched this query, so there is no evidence to answer from. "
    "This is a retrieval miss, not a finding of absence — try rephrasing, or widen "
    "the trial selection in the sidebar."
)
