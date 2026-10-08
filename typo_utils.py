"""
typo_utils.py

Generates realistic typos -- the kind a guest actually makes typing fast,
or that a voice-to-text engine produces from a mumbled word -- so we can:
  1. Augment training data with noisy versions (train the model to expect
     imperfect input, not just clean text).
  2. Build a separate "noisy" evaluation set to measure robustness
     specifically, instead of only reporting clean-text accuracy.

Typo types modeled (each is genuinely common, not made up for variety):
  - adjacent-key substitution (fat-fingering a nearby key on a QWERTY layout)
  - character deletion   (dropping a letter when typing fast)
  - character transposition (swapping two adjacent letters, e.g. "teh")
  - character duplication (holding a key too long)
"""

import random

QWERTY_ADJACENCY = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "rygf",
    "y": "tuhg", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def _typo_word(word: str) -> str:
    """Apply one random typo to a single word (words < 3 chars left alone,
    since typos there tend to just produce a different real word)."""
    if len(word) < 3:
        return word

    typo_type = random.choice(["substitute", "delete", "transpose", "duplicate"])
    i = random.randint(0, len(word) - 1)

    if typo_type == "substitute" and word[i] in QWERTY_ADJACENCY:
        replacement = random.choice(QWERTY_ADJACENCY[word[i]])
        return word[:i] + replacement + word[i + 1:]
    elif typo_type == "delete":
        return word[:i] + word[i + 1:]
    elif typo_type == "transpose" and i < len(word) - 1:
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    elif typo_type == "duplicate":
        return word[:i] + word[i] + word[i:]
    return word


def add_typos(text: str, word_error_prob: float = 0.35, max_words_affected: int = 2) -> str:
    """Introduce typos into a fraction of the words in a sentence. Only
    affects a capped number of words, since a real command is rarely
    mangled in every single word -- usually one or two get fumbled."""
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 3]
    random.shuffle(eligible)

    n_to_affect = min(max_words_affected, sum(1 for _ in eligible if random.random() < word_error_prob))
    for i in eligible[:n_to_affect]:
        words[i] = _typo_word(words[i])

    return " ".join(words)
