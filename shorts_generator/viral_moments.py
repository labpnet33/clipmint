"""Conservative, transcript-based completion checks for short-form clips.

No downloads, model calls, or audio classifiers. Uses only segment times and
explicit words/punctuation. A true laugh classifier would need acoustic data.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# One of these *at the very end* means a phrase is likely still being built.
# Keep the list conservative: e.g. 'hai' or 'tha' would be an unsafe Hindi cue.
_PENDING = frozenset({
    "and", "but", "because", "although", "while", "when", "if", "that",
    "so", "then", "to", "with", "for", "the", "a", "an", "which", "who",
    "aur", "lekin", "magar", "kyunki", "kyonki", "agar", "jab", "ki",
    "phir", "fir", "toh", "tab", "par", "isliye", "phirbhi",
})
# Words indicating an actual transcripted reaction, not a guessed sound peak.
_REACTION = re.compile(
    r"(?:\[|\(|\b)(?:laugh(?:ter|s|ing)?|giggl(?:e|es|ing)|"
    r"chuckl(?:e|es|ing)|applause|cheer(?:s|ing)?|clapping|"
    r"hahaha+|haha+|hehe+|lol)(?:\]|\)|\b)", re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"(?:[,;:]|\.{2,}|[\u2026\u2014\u2013-])\s*$")


def unfinished_phrase(text: str) -> bool:
    """Strong explicit sign of a cut-off clause; no punctuation guesswork."""
    s = str(text or "").strip()
    if not s:
        return False
    if _TRAILING_PUNCT.search(s):
        return True
    if s[-1] in '.!?।؟':
        return False
    terms = re.findall(r"[^\W_]+", s.lower(), re.UNICODE)
    return bool(terms and terms[-1] in _PENDING)


def reaction_cue(text: str) -> bool:
    return bool(_REACTION.search(str(text or "")))


def _valid_segments(transcript: Optional[Dict]) -> List[Dict]:
    result = []
    for seg in (transcript or {}).get("segments", []):
        if not isinstance(seg, dict):
            continue
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if 0 <= start < end:
            result.append({"start": start, "end": end,
                           "text": str(seg.get("text") or "")})
    return sorted(result, key=lambda x: x["start"])


def complete_punchline_end(
    segments: Sequence[Dict], start: float, chosen_end: float, hi: float,
) -> Tuple[float, List[str]]:
    """Extend only if an obvious unfinished clause or literal reaction fits.

    Keeps the user's length ceiling plus the existing 2.5-second grace; never
    cuts a new speech segment in half or moves a cut backwards. Deliberately
    does not infer laughter from amplitude alone.
    """
    notes: List[str] = []
    if not segments or chosen_end <= start:
        return chosen_end, notes
    ceiling = start + hi + 2.5
    # Include a segment ending just after a quiet-time snap (up to 0.45s).
    eligible = [(i, s) for i, s in enumerate(segments)
                if s["start"] < chosen_end - 0.05
                and s["end"] <= chosen_end + 0.45]
    if not eligible:
        return chosen_end, notes
    idx, tail = eligible[-1]
    if idx + 1 >= len(segments):
        return chosen_end, notes
    nxt = segments[idx + 1]
    gap = nxt["start"] - tail["end"]
    if not (-0.35 <= gap <= 1.6):
        return chosen_end, notes
    if not (chosen_end + 0.10 < nxt["end"] <= ceiling):
        return chosen_end, notes
    # Avoid long unrequested extensions for a short unfinished word.
    if nxt["end"] - chosen_end > 5.0:
        return chosen_end, notes
    incomplete = unfinished_phrase(tail.get("text", ""))
    reaction = reaction_cue(nxt.get("text", ""))
    if not incomplete and not reaction:
        return chosen_end, notes
    # If the next line is also evidently unfinished, stop rather than claiming
    # the following fragment is a payoff. This is a one-step conservative fix.
    if incomplete and unfinished_phrase(nxt.get("text", "")) and not reaction:
        return chosen_end, notes
    notes.append("included the rest of an unfinished line" if incomplete
                 else "kept a transcripted reaction after the payoff")
    return round(nxt["end"], 2), notes


def score_moment_integrity(highlight: Dict, transcript: Optional[Dict]) -> Dict:
    """Bounded score correction backed by actual segment evidence.

    A limited adjustment avoids overruling the existing model+audio+chat rank.
    Returns only numeric/short-label diagnostics, not spoken transcript text.
    """
    segments = _valid_segments(transcript)
    if not segments:
        return {"adjustment": 0, "reason": "no transcript evidence"}
    try:
        start, end = float(highlight.get("start_time", 0)), float(highlight.get("end_time", 0))
    except (TypeError, ValueError, OverflowError):
        return {"adjustment": 0, "reason": "invalid span"}
    if end <= start:
        return {"adjustment": 0, "reason": "invalid span"}
    inside = [s for s in segments if s["end"] > start and s["start"] < end]
    if not inside:
        return {"adjustment": 0, "reason": "no matching speech"}
    tail = inside[-1]
    # Genuine segment truncation: an end point falls more than 0.5s before
    # the end of a speech segment and is not a quiet cut after its conclusion.
    if tail["start"] < end - 0.05 and tail["end"] - end > 0.5:
        return {"adjustment": -10, "reason": "speech cut mid-segment"}
    following = next((s for s in segments if s["start"] >= end - 0.05
                      and s["start"] - end <= 1.6), None)
    if unfinished_phrase(tail.get("text", "")) and following:
        return {"adjustment": -8, "reason": "unfinished setup near clip end"}
    if any(reaction_cue(s["text"]) and s["end"] >= end - 4.0
           for s in inside):
        return {"adjustment": 3, "reason": "transcript confirms a reaction"}
    return {"adjustment": 0, "reason": "no clear issue"}
