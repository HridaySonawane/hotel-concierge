"""
synonym_utils.py

Addresses a DIFFERENT problem than typo_utils.py. Typos break word overlap
by mangling spelling ("warmer" -> "wamrer") -- character n-grams fix that
because the letters mostly still overlap. Paraphrases break word overlap
by using entirely different, correctly-spelled words ("cooler" vs "hot") --
character n-grams do NOT fix that, because there's no letter overlap to
find. "make this room a little cooler" and "it's too hot in here" share
almost no characters, let alone words.

The fix here is a normalization pass: map known synonym phrases to one
canonical form BEFORE vectorizing, for both training and inference. So
"power on", "kill", and "shut off" all collapse to one canonical token
before the model ever sees them -- which means "kill the tv" can now be
recognized as tv_off even though the model only ever saw "kill" used for
lighting during training, because after normalization both become the
same token.

This is a real, honest technique used in production NLU systems (often
called "canonicalization" or a "gazetteer" pass) -- but its coverage is
only as good as the dictionary. A synonym NOT in the list below still
won't be recognized; that's the genuine, remaining limitation, and it's
what a real embedding-based model would solve instead (see README).
"""

import re

# Each canonical key maps to phrases that should be normalized to it.
# Longer phrases are matched before shorter/single words to avoid partial
# overlaps (e.g. match "shut off" before a bare "shut").
#
# DELIBERATELY NOT covered here: temperature words (hot/warm/cold/cool)
# and lighting brightness words (bright/dark/dim). These are
# direction-ambiguous depending on sentence role: "it's too hot in here"
# wants a DECREASE, but "warmer" as an action word wants an INCREASE.
# The training data relies on "warmer" and "hot" being different tokens to
# tell increase from decrease apart -- collapsing them to one canonical
# form would make the model WORSE, not better, by erasing that exact
# distinction. (I only caught this by checking the actual training
# templates before wiring the dictionary in -- worth doing before you
# trust any synonym list, not just this one.)
SYNONYM_GROUPS = {
    "activate": ["turn on", "switch on", "power on"],
    "deactivate": ["turn off", "switch off", "power off", "shut off", "kill"],
    "quiet": ["quiet", "silence", "hush", "keep it down"],
}

# Build a single ordered list of (phrase, canonical) pairs, longest phrase
# first, so multi-word phrases match before their component words do.
_REPLACEMENTS = sorted(
    ((phrase, canonical) for canonical, phrases in SYNONYM_GROUPS.items()
     for phrase in phrases),
    key=lambda pair: len(pair[0]), reverse=True,
)


def normalize_synonyms(text: str) -> str:
    """Replace any known synonym phrase in `text` with its canonical form.
    Used as a preprocessing step for BOTH training and inference, so the
    model and the live input are always speaking the same normalized
    vocabulary."""
    result = text.lower()
    for phrase, canonical in _REPLACEMENTS:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        result = re.sub(pattern, canonical, result)
    return result
