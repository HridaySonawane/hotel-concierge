"""
tools.py

The concrete tools the agent can call. Each tool is deliberately simple
and mockable (no real hardware or ordering system exists here) but has a
REALISTIC failure mode built in on purpose -- a device that's offline, an
item that's out of stock -- because an agent that only ever sees success
isn't actually being tested as an agent. Handling failure gracefully
(retry, then escalate) is most of what separates "agent" from "script."

Two tools are not mocks: IntentTool loads the actual trained classifier
from the voice-intent-classifier project, and PreferenceTool loads the
actual output from the guest-preference-engine project. This is the
literal integration point mentioned in the project pitch -- the agent
orchestrates real components already built, not stand-ins for them.
"""

import random
import joblib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = Path(__file__).parent
RESOURCES_DIR = BASE_DIR / "resources"
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------------------
# Intent tool -- reuses the trained classifier from voice-intent-classifier
# ---------------------------------------------------------------------
class IntentTool:
    def __init__(self, confidence_threshold: float = 0.35):
        bundle = joblib.load(RESOURCES_DIR / "intent_model.joblib")
        self.pipeline = bundle["pipeline"]
        self.intent_to_action = bundle["intent_to_action"]
        self.confidence_threshold = confidence_threshold

    def classify(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx = proba.argmax()
        intent, confidence = classes[idx], float(proba[idx])

        if confidence < self.confidence_threshold:
            return {"text": text, "intent": "unclear", "confidence": round(confidence, 2),
                    "device": None, "action": None}

        device, action = self.intent_to_action.get(intent, (None, None))
        device = None if pd.isna(device) else device
        action = None if pd.isna(action) else action
        return {"text": text, "intent": intent, "confidence": round(confidence, 2),
                "device": device, "action": action}


# ---------------------------------------------------------------------
# Preference tool -- reuses output from guest-preference-engine
# ---------------------------------------------------------------------
class PreferenceTool:
    def __init__(self):
        self.predicted = pd.read_csv(DATA_DIR / "guest_predicted_setups.csv")
        self.guests = pd.read_csv(DATA_DIR / "guests.csv")

    def get_preferences(self, guest_id: str) -> Optional[dict]:
        row = self.predicted[self.predicted["guest_id"] == guest_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()


# ---------------------------------------------------------------------
# Device control tool -- mocked hardware, with realistic random failures
# ---------------------------------------------------------------------
@dataclass
class ToolResult:
    success: bool
    message: str
    detail: dict = field(default_factory=dict)


class DeviceControlTool:
    """Simulates sending a command to in-room hardware. ~8% of calls to
    thermostat/curtains fail (simulating a device offline / jammed motor)
    to force the agent to actually exercise its retry-and-escalate path
    rather than always taking the happy path."""

    FLAKY_DEVICES = {"thermostat", "curtains"}
    FAILURE_RATE = 0.08

    def execute(self, device: str, action: str, value: Optional[str] = None) -> ToolResult:
        if device in self.FLAKY_DEVICES and random.random() < self.FAILURE_RATE:
            return ToolResult(
                success=False,
                message=f"{device} did not respond (simulated hardware timeout).",
                detail={"device": device, "action": action},
            )
        if device == "television" and action == "turn_on" and value:
            return ToolResult(
                success=True,
                message=f"television set to play '{value}'.",
                detail={"device": device, "action": action, "value": value},
            )
        value_str = f" to {value}" if value else ""
        return ToolResult(
            success=True,
            message=f"{device} set to '{action}'{value_str}.",
            detail={"device": device, "action": action, "value": value},
        )


# ---------------------------------------------------------------------
# Room service tool -- mocked ordering system
# ---------------------------------------------------------------------
class RoomServiceTool:
    MENU = {
        "snacks": ["mixed nuts", "cheese platter", "chocolate assortment"],
        "coffee": ["espresso", "cappuccino", "black coffee"],
        "breakfast": ["continental breakfast", "eggs benedict", "pancakes"],
    }
    OUT_OF_STOCK = {"cheese platter"}  # simulated unavailable item
    DIETARY_OPTIONS = {
        "mixed nuts": {"vegetarian", "vegan", "non_vegetarian"},
        "cheese platter": {"vegetarian", "non_vegetarian"},
        "chocolate assortment": {"vegetarian", "vegan", "non_vegetarian"},
        "continental breakfast": {"vegetarian", "vegan", "non_vegetarian"},
        "eggs benedict": {"vegetarian", "non_vegetarian"},
        "pancakes": {"vegetarian", "non_vegetarian"},
        "espresso": {"vegetarian", "vegan", "non_vegetarian"},
        "cappuccino": {"vegetarian", "non_vegetarian"},
        "black coffee": {"vegetarian", "vegan", "non_vegetarian"},
    }

    def order(self, category: str, dining_pref: Optional[str] = None) -> ToolResult:
        items = self.MENU.get(category)
        if not items:
            return ToolResult(success=False, message=f"No menu category matching '{category}'.")

        available = [i for i in items if i not in self.OUT_OF_STOCK]
        if dining_pref:
            available = [i for i in available
                         if dining_pref in self.DIETARY_OPTIONS.get(i, set())]
        if not available:
            preference_text = f" for a {dining_pref} preference" if dining_pref else ""
            return ToolResult(success=False,
                              message=f"No {category} items are available{preference_text}.")

        chosen = available[0]
        return ToolResult(success=True, message=f"Order placed: {chosen}.",
                           detail={"category": category, "item": chosen,
                                   "dining_pref": dining_pref})

    def order_for_preference(self, dining_pref: str) -> ToolResult:
        """Choose a default meal that satisfies the guest's dining preference."""
        for category in ("breakfast", "snacks"):
            result = self.order(category, dining_pref)
            if result.success:
                return result
        return ToolResult(success=False,
                          message=f"No room-service meal matches a {dining_pref} preference.")


# ---------------------------------------------------------------------
# Staff escalation tool -- logs an escalation instead of executing
# ---------------------------------------------------------------------
class EscalateStaffTool:
    def __init__(self):
        self.log = []

    def escalate(self, reason: str, priority: str, original_text: str) -> ToolResult:
        entry = {"reason": reason, "priority": priority, "original_text": original_text}
        self.log.append(entry)
        return ToolResult(success=True, message=f"Escalated to staff ({priority} priority): {reason}",
                           detail=entry)
