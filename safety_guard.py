"""
safety_guard.py

The "abnormal input" layer. An agent that can actually control physical
hotel room hardware (locks, DND, room service) needs to check what it's
being asked BEFORE it plans or executes anything -- a device-control
intent classifier alone will happily try to classify "there's smoke
coming from the vent" as some kind of command, because that's all it
knows how to do. This module's whole job is to catch the cases where
"classify it as a device command" is the wrong response entirely.

Four categories of abnormal input, each handled differently because they
need different responses:

1. EMERGENCY   -- immediate safety risk (fire, gas, medical, intrusion).
                  Must bypass all normal planning and escalate instantly,
                  with no delay for classification or confirmation.
2. INJECTION   -- attempts to manipulate the agent into doing something
                  outside its authority (unlocking doors, revealing guest
                  data, ignoring its own rules). Must be refused, not
                  "interpreted charitably."
3. GIBBERISH   -- input with no real linguistic content (mashed keyboard,
                  empty string, pure noise). Should prompt a rephrase, not
                  be forced through the classifier to get a low-confidence
                  guess.
4. CONFLICT    -- a multi-part request whose own sub-goals contradict each
                  other (e.g. "turn the lights on and off"). Caught AFTER
                  planning, once sub-intents are known.

Each check is intentionally simple and auditable (keyword/pattern based)
rather than another ML model -- for a safety-critical gate like this,
being able to point at the exact rule that fired matters more than
squeezing out extra accuracy from a black box.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------
# 1. Emergency detection
# ---------------------------------------------------------------------
# Deliberately broad and biased toward false positives over false
# negatives -- for genuine emergencies, escalating unnecessarily costs a
# staff member a wasted trip; failing to escalate could cost much more.
EMERGENCY_PATTERNS = [
    r"\bfire\b", r"\bsmoke\b", r"\bburning\b", r"\bsmell.*gas\b", r"\bgas leak\b",
    r"\bcan'?t breathe\b", r"\bchest pain\b", r"\bheart attack\b", r"\bunconscious\b",
    r"\bcollapsed\b", r"\bbleeding\b", r"\bseizure\b", r"\boverdose\b",
    r"\bintruder\b", r"\bbreak.?in\b", r"\bsomeone.*(trying to|broke).*(door|room)\b",
    r"\bhelp me\b", r"\bemergency\b", r"\bcall (an ambulance|911|999|112)\b",
    r"\bflooding\b", r"\bwater everywhere\b", r"\belectric shock\b",
]
_EMERGENCY_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------
# 2. Prompt-injection / out-of-authority detection
# ---------------------------------------------------------------------
# Two sub-patterns: attempts to override the agent's own instructions,
# and requests for actions the agent should never take regardless of
# phrasing (unlocking doors remotely, disabling security, revealing other
# guests' data).
INJECTION_INSTRUCTION_PATTERNS = [
    r"\bignore (your|all|previous|the) instructions\b",
    r"\byou are now\b", r"\bact as\b.*\b(admin|root|developer|hacker)\b",
    r"\breveal your (system )?prompt\b", r"\bwhat are your instructions\b",
    r"\bpretend (you|to be)\b", r"\bdisregard\b.*\b(rules|policy|guidelines)\b",
    r"\boverride\b.*\b(safety|security|lock)\b",
]
OUT_OF_AUTHORITY_PATTERNS = [
    r"\bunlock (the |all )?(door|room)s?\b", r"\bdisable (the )?(security|alarm|camera)\b",
    r"\bgive me (the )?(master key|admin access|override code)\b",
    r"\b(other|another) guest'?s? (room|data|information|details)\b",
    r"\bcheck (in|out) .* without\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_INSTRUCTION_PATTERNS), re.IGNORECASE)
_AUTHORITY_RE = re.compile("|".join(OUT_OF_AUTHORITY_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------
# 3. Gibberish detection
# ---------------------------------------------------------------------
_COMMON_ENGLISH_WORDS = {
    "the", "a", "an", "i", "you", "please", "can", "could", "would", "to",
    "in", "on", "at", "it", "is", "are", "my", "me", "and", "for", "of",
    "room", "light", "lights", "tv", "music", "please", "want", "need",
    "turn", "make", "set", "some", "this", "that", "here", "up", "down",
    "with", "help", "not", "sure", "what", "do", "get", "have", "has",
    # common short interjections/abbreviations that are real but
    # consonant-heavy or vowel-free -- would otherwise false-positive
    # against the implausible-word heuristic below
    "hmm", "huh", "hey", "hi", "yes", "no", "yeah", "nope", "brb", "ok",
    "okay", "thanks", "thank", "sorry", "um", "uh",
}


def _is_implausible_word(word: str) -> bool:
    """A word is treated as implausible English if it has no vowels, is a
    single character repeated 3+ times (keyboard mashing like "zzz" or
    "aaaa"), or contains a run of 4+ consecutive consonants -- real
    English words essentially never do this ("strengths" has a 3-run at
    most). This catches random keyboard mashing that a naive
    vowel-presence check misses, since mashed text often still contains
    a stray vowel by chance (e.g. "asdkjfh" has an 'a')."""
    w = word.lower()
    if len(w) > 12:
        return False  # long words we don't judge -- avoid false positives

    if len(set(w)) == 1 and len(w) >= 3:
        return True  # "zzz", "aaaa"

    if not re.search(r"[aeiou]", w):
        return len(w) > 2  # no vowels at all (allow short real words like "tv")

    max_consonant_run = 0
    current_run = 0
    for ch in w:
        if ch not in "aeiou":
            current_run += 1
            max_consonant_run = max(max_consonant_run, current_run)
        else:
            current_run = 0
    return max_consonant_run >= 4


def _looks_like_gibberish(text: str) -> bool:
    """Flags input with essentially no recognizable English content --
    empty strings, pure keyboard mashing, or extremely short fragments.
    Deliberately conservative: only fires when there's close to nothing
    for the classifier to work with, since a false positive here just
    means asking the guest to repeat themselves once."""
    stripped = text.strip()
    if len(stripped) < 2:
        return True

    words = re.findall(r"[a-zA-Z']+", stripped.lower())
    if not words:
        return True  # no letters at all -- e.g. "12345" or "!!!"

    known = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
    if known > 0:
        return False  # at least one recognizable word -- treat as real input

    plausible = sum(1 for w in words if not _is_implausible_word(w))
    return plausible < max(1, len(words)) * 0.5


# ---------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------
@dataclass
class GuardResult:
    flagged: bool
    category: Optional[str]  # "emergency" | "injection" | "gibberish" | None
    message: str
    matched_rule: Optional[str] = None


def check_input(text: str) -> GuardResult:
    """Run all input-level checks, in priority order. Emergency is checked
    first and always wins -- even a message that ALSO looks like an
    injection attempt or is partly garbled should still escalate
    immediately if it contains an emergency signal."""

    emergency_match = _EMERGENCY_RE.search(text)
    if emergency_match:
        return GuardResult(
            flagged=True, category="emergency",
            message="Detected a possible safety emergency -- escalating to "
                     "hotel staff immediately instead of treating this as a "
                     "device command.",
            matched_rule=emergency_match.group(0),
        )

    injection_match = _INJECTION_RE.search(text) or _AUTHORITY_RE.search(text)
    if injection_match:
        return GuardResult(
            flagged=True, category="injection",
            message="This request asks for something outside what a room "
                     "assistant is authorized to do. Declining and logging "
                     "it for staff review rather than attempting it.",
            matched_rule=injection_match.group(0),
        )

    if _looks_like_gibberish(text):
        return GuardResult(
            flagged=True, category="gibberish",
            message="Couldn't make sense of that -- asking the guest to "
                     "rephrase rather than guessing.",
        )

    return GuardResult(flagged=False, category=None, message="")


def check_conflicts(steps: list) -> Optional[str]:
    """Given a list of planned steps (each a dict with 'device' and
    'action' keys), detect direct contradictions -- the same device
    targeted with opposite actions in the same request. Returns a
    human-readable conflict description, or None if no conflict found."""
    OPPOSITES = {
        "turn_on": "turn_off", "turn_off": "turn_on",
        "increase": "decrease", "decrease": "increase",
        "dim": "brighten", "brighten": "dim",
        "play": "stop", "stop": "play",
        "open": "close", "close": "open",
        "enable": "disable", "disable": "enable",
    }

    seen = {}  # device -> action
    for step in steps:
        device, action = step.get("device"), step.get("action")
        if not device or not action:
            continue
        if device in seen:
            prev_action = seen[device]
            if OPPOSITES.get(prev_action) == action:
                return (f"Conflicting instructions for '{device}': asked to "
                        f"'{prev_action}' and '{action}' in the same request.")
        seen[device] = action
    return None
