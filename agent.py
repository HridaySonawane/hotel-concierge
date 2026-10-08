"""
agent.py

The orchestrator. This is what makes the difference between "a script
that calls a classifier" and an agent: it holds state across turns, plans
multi-step goals, executes with retry-then-escalate rather than
fail-silently, and gates everything through a safety layer before any of
that happens.

Loop, per request:
  1. GUARD    -- safety_guard.check_input(). Emergencies, injection
                 attempts, and gibberish are caught here and handled
                 completely differently from a normal request (see
                 safety_guard.py for why each needs its own response).
  2. PLAN     -- planner.plan() decomposes the request into concrete
                 steps (device + action), using scene templates or
                 per-clause intent classification.
  3. CHECK    -- safety_guard.check_conflicts() catches self-contradictory
                 plans (e.g. "turn the lights on and off") before
                 anything is executed.
  4. EXECUTE  -- each step is run through the matching tool. Failures are
                 retried once (simulating "maybe that was a transient
                 glitch"), and if they still fail, escalated to staff
                 instead of silently giving up or pretending to succeed.
  5. MEMORY   -- every request and its outcome is appended to that guest's
                 session history, so a later request in the same session
                 could reference earlier context (this project keeps that
                 simple -- see README for where a real system would go
                 further).
"""

from dataclasses import dataclass, field
from typing import Optional
from tools import IntentTool, PreferenceTool, DeviceControlTool, RoomServiceTool, EscalateStaffTool
from planner import Planner
import safety_guard


@dataclass
class AgentResponse:
    guest_message: str
    status: str  # "completed" | "partial" | "escalated" | "refused" | "clarification_needed"
    steps: list = field(default_factory=list)
    escalations: list = field(default_factory=list)
    guard_flag: Optional[str] = None


class ConciergeAgent:
    def __init__(self):
        self.intent_tool = IntentTool()
        self.preference_tool = PreferenceTool()
        self.device_tool = DeviceControlTool()
        self.room_service_tool = RoomServiceTool()
        self.escalation_tool = EscalateStaffTool()
        self.planner = Planner(self.intent_tool)
        self.sessions = {}  # guest_id -> list of past AgentResponse-like records

    def _remember(self, guest_id: str, text: str, response: AgentResponse):
        self.sessions.setdefault(guest_id, []).append({"text": text, "response": response})

    def handle_request(self, guest_id: str, text: str) -> AgentResponse:
        # ---- 1. Input-level safety guard ----
        guard_result = safety_guard.check_input(text)
        if guard_result.flagged:
            if guard_result.category == "emergency":
                esc = self.escalation_tool.escalate(
                    reason=f"Possible emergency: matched '{guard_result.matched_rule}'",
                    priority="URGENT", original_text=text,
                )
                response = AgentResponse(
                    guest_message=("This sounds like it could be an emergency. I've alerted "
                                    "hotel staff immediately -- help is on the way. If this is "
                                    "life-threatening, please also call local emergency services directly."),
                    status="escalated", escalations=[esc.detail], guard_flag="emergency",
                )
            elif guard_result.category == "injection":
                esc = self.escalation_tool.escalate(
                    reason=f"Out-of-scope/authority request: matched '{guard_result.matched_rule}'",
                    priority="normal", original_text=text,
                )
                response = AgentResponse(
                    guest_message=("I'm not able to help with that request. I've noted it for "
                                    "staff review in case you need something I'm not authorized to do."),
                    status="refused", escalations=[esc.detail], guard_flag="injection",
                )
            else:  # gibberish
                response = AgentResponse(
                    guest_message="Sorry, I didn't catch that -- could you rephrase?",
                    status="clarification_needed", guard_flag="gibberish",
                )
            self._remember(guest_id, text, response)
            return response

        # ---- 2. Plan ----
        steps = self.planner.plan(text)

        # ---- 3. Conflict check on the plan itself ----
        conflict = safety_guard.check_conflicts(steps)
        if conflict:
            response = AgentResponse(
                guest_message=f"I want to make sure I get this right -- {conflict} "
                               f"Could you clarify what you'd like?",
                status="clarification_needed", steps=steps,
            )
            self._remember(guest_id, text, response)
            return response

        # ---- 4. Execute each step, with retry-then-escalate ----
        executed_steps = []
        escalations = []
        needs_clarification = []

        for step in steps:
            if step["source"] == "unclear":
                needs_clarification.append(step)
                step["result"] = "unclear"
                executed_steps.append(step)
                continue

            if step["device"] == "room_service":
                category = step.get("value")
                if category:
                    preferences = self.preference_tool.get_preferences(guest_id)
                    dining_pref = preferences.get("dining_pref") if preferences else None
                    result = self.room_service_tool.order(category, dining_pref)
                elif step.get("action") == "request":
                    preferences = self.preference_tool.get_preferences(guest_id)
                    dining_pref = preferences.get("dining_pref") if preferences else None
                    if dining_pref:
                        result = self.room_service_tool.order_for_preference(dining_pref)
                        step["source"] = "room_service:preference"
                        step["action"] = "order"
                        step["value"] = dining_pref
                    else:
                        esc = self.escalation_tool.escalate(
                            reason="Room service requested but no specific item or dining preference "
                                   "detected -- needs staff follow-up.",
                            priority="normal", original_text=text,
                        )
                        step["result"] = "escalated"
                        step["result_message"] = "Forwarded to staff to confirm what you'd like."
                        escalations.append(esc.detail)
                        executed_steps.append(step)
                        continue
                    if result is not None and not result.success:
                        step["result"] = "failed"
                        step["result_message"] = result.message
                        esc = self.escalation_tool.escalate(
                            reason=f"Room service preference could not be fulfilled: {result.message}",
                            priority="normal", original_text=text,
                        )
                        escalations.append(esc.detail)
                        executed_steps.append(step)
                        continue
                else:
                    # Reached via the generic intent classifier (e.g. "can I
                    # get some room service") rather than a specific keyword
                    # match (snack/coffee/breakfast) -- we know they want
                    # room service but not WHAT, so escalate to a human
                    # rather than guessing an item on the guest's behalf.
                    esc = self.escalation_tool.escalate(
                        reason="Room service requested but no specific item "
                               "detected -- needs staff follow-up.",
                        priority="normal", original_text=text,
                    )
                    step["result"] = "escalated"
                    step["result_message"] = "Forwarded to staff to confirm what you'd like."
                    escalations.append(esc.detail)
                    executed_steps.append(step)
                    continue
            elif step["device"] and step["action"]:
                result = self.device_tool.execute(step["device"], step["action"], step.get("value"))
                if not result.success:
                    # retry once before giving up
                    result = self.device_tool.execute(step["device"], step["action"], step.get("value"))
            else:
                result = None

            if result is None:
                step["result"] = "skipped"
            elif result.success:
                step["result"] = "success"
                if step.get("device") == "wake_up_service" and step.get("action") == "schedule":
                    wake_time = step.get("value")
                    step["result_message"] = f"Wake-up call scheduled{f' for {wake_time}' if wake_time else ''}."
                else:
                    step["result_message"] = result.message
            else:
                step["result"] = "failed"
                step["result_message"] = result.message
                esc = self.escalation_tool.escalate(
                    reason=f"Device action failed after retry: {result.message}",
                    priority="normal", original_text=text,
                )
                escalations.append(esc.detail)

            executed_steps.append(step)

        # ---- 5. Summarize ----
        n_success = sum(1 for s in executed_steps if s.get("result") == "success")
        n_failed = sum(1 for s in executed_steps if s.get("result") == "failed")
        n_escalated_steps = sum(1 for s in executed_steps if s.get("result") == "escalated")
        n_unclear = len(needs_clarification)
        n_issues = n_failed + n_escalated_steps

        parts = []
        if n_success:
            successful = [s for s in executed_steps if s.get("result") == "success"]
            movie_steps = [s for s in successful if s.get("source") == "inferred:movie_setup"]
            movie_tv = next((s for s in movie_steps if s.get("device") == "television"), None)
            movie_ready = bool(
                movie_tv
                and any(s.get("device") == "lighting" and s.get("action") == "dim" for s in successful)
                and any(s.get("device") == "curtains" and s.get("action") == "close" for s in successful)
            )
            if movie_ready:
                title = movie_tv.get("value") or "your movie"
                parts.append(f"I've set up the room to watch {title}: lights dimmed, curtains closed, and television on.")

            vague_action_copy = {
                ("thermostat", "increase"): "I've warmed the room.",
                ("thermostat", "decrease"): "I've cooled the room.",
                ("lighting", "brighten"): "I've brightened the room.",
                ("lighting", "dim"): "I've softened the lighting.",
                ("curtains", "close"): "I've closed the curtains for privacy.",
                ("curtains", "open"): "I've opened the curtains for daylight.",
            }
            done = []
            for step in successful:
                is_movie_setup_action = (
                    step.get("device") == "lighting" and step.get("action") == "dim"
                ) or (
                    step.get("device") == "curtains" and step.get("action") == "close"
                )
                if movie_ready and (step in movie_steps or is_movie_setup_action):
                    continue
                if step.get("source", "").startswith("inferred:"):
                    done.append(vague_action_copy.get((step.get("device"), step.get("action")), f"{step['device']} {step['action']}"))
                elif step.get("device") == "wake_up_service" and step.get("action") == "schedule":
                    wake_time = step.get("value")
                    done.append(f"I've scheduled a wake-up call{f' for {wake_time}' if wake_time else ''}")
                elif step.get("device") == "television" and step.get("value"):
                    done.append(f"I've set the television to play {step['value']}.")
                else:
                    done.append(f"{step['device']} {step['action']}")
            if done:
                parts.append(f"Done: {', '.join(done)}.")
        if n_issues:
            parts.append(f"{n_issues} of your requests need staff follow-up, and I've alerted them.")
        if n_unclear:
            unclear_texts = [f"\"{s['text']}\"" for s in needs_clarification]
            parts.append(f"I wasn't sure what you meant by {', '.join(unclear_texts)} -- could you clarify?")

        guest_message = " ".join(parts) if parts else "Sorry, I couldn't process that request."

        if n_unclear and not n_success and not n_issues:
            status = "clarification_needed"
        elif n_issues and n_success:
            status = "partial"
        elif n_issues:
            status = "escalated"
        else:
            status = "completed"

        response = AgentResponse(
            guest_message=guest_message, status=status,
            steps=executed_steps, escalations=escalations,
        )
        self._remember(guest_id, text, response)
        return response

    def get_session_history(self, guest_id: str) -> list:
        return self.sessions.get(guest_id, [])
