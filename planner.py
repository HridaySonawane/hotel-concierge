"""
planner.py

Turns one free-text guest request -- possibly covering several goals at
once -- into an ordered list of concrete steps the agent can execute and
verify one at a time. This is the "planning" part of the agent loop.

Two planning strategies, combined:

1. SCENE TEMPLATES -- common multi-device requests ("movie night", "good
   night") map to a known sequence of device actions. This is genuinely
   how a lot of real smart-hotel systems work (DigiValet's own product
   pitch is built around named "scenes"), and it means the agent doesn't
   have to reinvent the plan for well-known patterns every time.
2. CLAUSE-LEVEL INTENT CLASSIFICATION -- for anything not covered by a
   known scene, the request is split on conjunctions ("and", "then",
   ";", "also") and each clause is run through the existing
   voice-intent-classifier model. This handles the long tail of
   requests that don't match a named scene.

No LLM is used for planning here -- see README for why (this sandbox
can't reach an LLM API), and for the code showing how you'd swap an
LLM-based planner in if you had one available.
"""

import re
from tools import IntentTool

SCENE_TEMPLATES = {
    "movie night": [
        {"device": "lighting", "action": "dim"},
        {"device": "curtains", "action": "close"},
        {"device": "television", "action": "turn_on"},
    ],
    "date night": [
        {"device": "lighting", "action": "dim"},
        {"device": "audio", "action": "play"},
    ],
    "romantic": [
        {"device": "lighting", "action": "dim"},
        {"device": "audio", "action": "play"},
    ],
    "good night": [
        {"device": "lighting", "action": "turn_off"},
        {"device": "curtains", "action": "close"},
        {"device": "dnd", "action": "enable"},
    ],
    "going to sleep": [
        {"device": "lighting", "action": "turn_off"},
        {"device": "curtains", "action": "close"},
        {"device": "dnd", "action": "enable"},
    ],
    "good morning": [
        {"device": "curtains", "action": "open"},
        {"device": "lighting", "action": "brighten"},
    ],
    "waking up": [
        {"device": "curtains", "action": "open"},
        {"device": "lighting", "action": "brighten"},
    ],
    "heading out": [
        {"device": "lighting", "action": "turn_off"},
        {"device": "television", "action": "turn_off"},
    ],
    "leaving the room": [
        {"device": "lighting", "action": "turn_off"},
        {"device": "television", "action": "turn_off"},
    ],
    "relaxation": [
        {"device": "lighting", "action": "dim"},
        {"device": "audio", "action": "play"},
    ],
    "focus mode": [
        {"device": "lighting", "action": "brighten"},
        {"device": "audio", "action": "stop"},
        {"device": "dnd", "action": "enable"},
    ],
    "reading": [
        {"device": "lighting", "action": "brighten"},
        {"device": "audio", "action": "stop"},
    ],
    "privacy": [
        {"device": "curtains", "action": "close"},
        {"device": "dnd", "action": "enable"},
    ],
}

# Match natural-language goals to small, reversible hotel workflows. Keep
# these ahead of the general classifier: it can otherwise turn vague goals
# like “I want to relax” into unrelated device actions.
WORKFLOW_ALIASES = [
    (re.compile(r"\b(?:get (?:me )?ready for bed|getting ready for bed|go to bed|going to bed|turning in|want to sleep|need to sleep|go to sleep|good night|wind down for bed|settle in for the night|get some rest|prepare for a restful night)\b", re.I), "good night"),
    (re.compile(r"\b(?:good morning|start my day|ready for the day|time to get ready)\b", re.I), "good morning"),
    (re.compile(r"\b(?:relax|unwind|wind down|chill(?: out)?|take it easy|quiet evening|peaceful evening|cozy evening|make (?:it|the room) (?:cozy|cozier))\b", re.I), "relaxation"),
    (re.compile(r"\b(?:focus|work|study|get (?:some )?work done|concentrate)\b", re.I), "focus mode"),
    (re.compile(r"\b(?:read|reading)\b", re.I), "reading"),
    (re.compile(r"\b(?:privacy|leave me alone|left alone|some space)\b", re.I), "privacy"),
]

ROOM_SERVICE_KEYWORDS = {
    "snacks": ["snack", "nuts", "cheese platter", "chocolate", "chips", "crisps", "cookie", "dessert", "something to eat", "bite to eat"],
    "coffee": ["coffee", "espresso", "cappuccino"],
    "breakfast": ["breakfast", "eggs", "pancakes"],
}

VAGUE_REQUEST_RULES = [
    (re.compile(r"\b(?:i(?:['’]m| am| feel) (?:a bit )?(?:freezing|cold|chilly)|(?:it['’]s|it is) (?:a bit |quite |really |too )?(?:freezing|cold|chilly)|room is (?:freezing|cold|chilly)|(?:room |it )?feels cold|getting chilly|could use some warmth|(?:a bit|a touch) warmer|make (?:the room|it) warmer)\b", re.I), "thermostat", "increase"),
    (re.compile(r"\b(?:i(?:'m| am) hot|(?:it's|it is) too hot|too warm in here|room feels warm|could use some cool air|make it cooler)\b", re.I), "thermostat", "decrease"),
    (re.compile(r"\b(?:too dark|so dark|room is dark|can['’]t see(?: anything| a thing)?|cannot see(?: anything| a thing)?|hard to see(?: anything)?|need more light)\b", re.I), "lighting", "brighten"),
    (re.compile(r"\b(?:too bright|light is harsh|lights are harsh|so much light|need softer light)\b", re.I), "lighting", "dim"),
    (re.compile(r"\b(?:need|want|would like) (?:a little |a bit of |some )?privacy|could use some privacy|feel exposed|people can see in\b", re.I), "curtains", "close"),
    (re.compile(r"\b(?:need|want|would like) (?:some )?(?:daylight|sunlight)|could use some daylight\b", re.I), "curtains", "open"),
    (re.compile(r"\b(?:make (?:it|the room) brighter|more light in here|brighten (?:it|the room))\b", re.I), "lighting", "brighten"),
    (re.compile(r"\b(?:make (?:it|the room) darker|less light in here|dim (?:it|the room))\b", re.I), "lighting", "dim"),
    (re.compile(r"\b(?:i(?:'m| am) hungry|could use (?:a bite|something to eat)|need something to eat|i want to eat)\b", re.I), "room_service", "request"),
]

CLAUSE_SPLIT_RE = re.compile(r"\band\b|\bthen\b|\balso\b|[;,]", re.IGNORECASE)


class Planner:
    def __init__(self, intent_tool: IntentTool):
        self.intent_tool = intent_tool

    def _match_scene(self, text: str):
        text_lower = text.lower()
        for phrase, template in SCENE_TEMPLATES.items():
            if phrase in text_lower:
                return phrase, template
        for pattern, phrase in WORKFLOW_ALIASES:
            match = pattern.search(text)
            if match and not re.search(r"\b(?:don't|do not|not|never|without)\s+(?:really\s+)?$", text[:match.start()], re.I):
                return phrase, SCENE_TEMPLATES[phrase]
        return None, None

    def _movie_title(self, text: str):
        """Infer a movie setup from a watch request, even without 'movie night'."""
        movie_clause = re.split(r"\band\b|\bthen\b|\balso\b|[;,]", text.strip(), maxsplit=1, flags=re.IGNORECASE)[0]
        match = re.search(
            r"\b(?:watch|put on|play|stream|see)\s+(?:the\s+)?(.+?)"
            r"(?:\s+(?:tonight|today|this evening|later))?[.!?]*$",
            movie_clause, re.IGNORECASE,
        )
        if not match:
            return None

        requested = match.group(1).strip(" .?!")
        if re.fullmatch(r"(?:a |the )?(?:movie|film)", requested, re.IGNORECASE):
            return requested
        if re.fullmatch(r"(?:tv|television|the game|game|sports|news|(?:video|board) games?|(?:some )?music|(?:a |some )?songs?|radio|podcast|my playlist)", requested, re.IGNORECASE):
            return None
        if not re.search(r"\b(movie|film)\b", requested, re.IGNORECASE):
            words = requested.split()
            has_single_word_title = len(words) == 1 and words[0][:1].isupper()
            if len(words) < 2 and not has_single_word_title:
                return None
        return requested

    def _match_room_service(self, text: str):
        text_lower = text.lower()
        for category, keywords in ROOM_SERVICE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return None

    def _is_generic_food_request(self, text: str):
        return bool(re.search(r"\b(?:hungry|starving|peckish|something to eat|bite to eat|could eat|need food|want to eat)\b", text, re.I))

    def _infer_vague_request(self, text: str):
        """Map clear indirect phrasing to a safe, reversible room action."""
        for pattern, device, action in VAGUE_REQUEST_RULES:
            if pattern.search(text):
                return device, action
        return None

    def plan(self, text: str) -> list:
        """Map each request clause to an ordered plan before execution."""
        clauses = [clause.strip() for clause in CLAUSE_SPLIT_RE.split(text) if clause.strip()]
        if not clauses:
            clauses = [text]

        steps = []

        def add_step(clause, source, device, action, confidence, value=None, intent=None):
            # Avoid repeating the same requested operation while preserving
            # opposite operations so the agent can detect and clarify conflicts.
            if any(s.get("device") == device and s.get("action") == action for s in steps):
                return
            step = {
                "step_id": len(steps) + 1, "source": source, "text": clause,
                "device": device, "action": action, "confidence": confidence,
            }
            if value is not None:
                step["value"] = value
            if intent is not None:
                step["intent"] = intent
            steps.append(step)

        for clause in clauses:
            handled = False

            scene_phrase, scene_template = self._match_scene(clause)
            if scene_template:
                for action_spec in scene_template:
                    add_step(clause, f"scene:{scene_phrase}", action_spec["device"],
                             action_spec["action"], 1.0)
                handled = True

            movie_title = self._movie_title(clause)
            if movie_title:
                for device, action in (
                    ("lighting", "dim"),
                    ("curtains", "close"),
                    ("television", "turn_on"),
                ):
                    add_step(clause, "inferred:movie_setup", device, action, 0.88,
                             movie_title if device == "television" else None)
                handled = True

            room_service_category = self._match_room_service(clause)
            if room_service_category:
                add_step(clause, "room_service", "room_service", "order", 1.0,
                         room_service_category)
                handled = True
            elif self._is_generic_food_request(clause):
                add_step(clause, "intent", "room_service", "request", 0.86)
                handled = True

            inferred = self._infer_vague_request(clause)
            if inferred:
                device, action = inferred
                add_step(clause, f"inferred:{device}_{action}", device, action, 0.86)
                handled = True

            if handled:
                continue

            result = self.intent_tool.classify(clause)
            # The classifier is useful for familiar commands but can be
            # overconfident on open-ended language. Ask for clarification
            # rather than executing a weak or unsupported prediction.
            if result["intent"] != "unclear" and result["confidence"] < 0.5:
                result = {**result, "intent": "unclear", "device": None, "action": None}
            # The model's channel-change label has no matching device tool;
            # the safe supported interpretation of a watch request is to
            # turn the television on.
            if result.get("device") == "television" and result.get("action") == "change_channel":
                result = {**result, "action": "turn_on"}
            value = None
            if result.get("device") == "wake_up_service":
                alarm_time = re.search(r"\b(?:at\s*)?(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)\b", clause, re.I)
                if alarm_time:
                    value = alarm_time.group(1).replace(" ", "")

            add_step(clause,
                     "intent" if result["intent"] != "unclear" else "unclear",
                     result["device"], result["action"], result["confidence"],
                     value, result.get("intent"))

        return steps
