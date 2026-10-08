# Hotel Concierge Agent

An agent that takes a free-text guest request — possibly covering several
goals at once, like _"set up the room for movie night and order some
snacks"_ — plans the steps needed, executes them against (mocked) hotel
systems, retries and escalates failures instead of silently giving up, and
gates everything through a safety layer that catches emergencies,
manipulation attempts, and nonsense input before any of that happens.

This is the fourth project in the DigiValet series and deliberately
**reuses the first two as components rather than rebuilding them**: the
intent classifier from `voice-intent-classifier` and the guest data from
`guest-preference-engine` are loaded directly as tools the agent calls,
not reimplemented. The point of this project is the orchestration layer
on top, not redoing work already done.

The planner maps common guest goals into ordered workflows (for example,
getting ready for bed, unwinding while putting on a movie, preparing to focus,
or asking for privacy). It handles each part of a multi-part request in order;
when the intent is weak or the plan conflicts, it asks the guest to clarify.

## Why this counts as "agentic" and not just "a script that calls a model"

Four things distinguish it, each with a corresponding file:

| Trait                                                                                           | Where                         |
| ----------------------------------------------------------------------------------------------- | ----------------------------- |
| **Plans** multi-step goals, not just single commands                                            | `planner.py`                  |
| **Uses tools** autonomously (device control, room service, escalation)                          | `tools.py`                    |
| **Maintains state** across turns per guest                                                      | `agent.py` (`self.sessions`)  |
| **Self-corrects** — retries failures, escalates what it can't fix, refuses to guess when unsafe | `agent.py`, `safety_guard.py` |

## Architecture: the request loop

```
guest text
    │
    ▼
[1] SAFETY GUARD  ──── emergency? ────────► escalate immediately, stop
    │                  injection attempt? ─► refuse, log, stop
    │                  gibberish? ─────────► ask to rephrase, stop
    ▼ (clean input)
[2] PLAN          ──── scene template match (e.g. "movie night")
    │                  OR per-clause intent classification
    ▼
[3] CONFLICT CHECK ─── contradictory sub-goals? ──► ask for clarification
    ▼ (no conflicts)
[4] EXECUTE        ─── use guest preferences when they can resolve the request
    │                  then try → fail? retry once → still fail? escalate
    ▼
[5] SUMMARIZE      ─── natural-language response + structured trace
    │
    ▼
[6] REMEMBER       ─── append to this guest's session history
```

## Run it

```bash
pip install scikit-learn pandas numpy joblib streamlit
python demo_scenarios.py      # safety guard evaluation + scripted scenarios
streamlit run dashboard.py    # live chat interface
```

### Next.js application

The `app/` directory contains the Next.js concierge interface. Its chat API
calls the same Python `ConciergeAgent` used by Streamlit, so the safety guard,
trained intent classifier, preference data, mocked tools, retries, and
escalations come from the project implementation rather than a second copy of
the rules. Device actions and room service remain simulated; no real hotel
systems are connected. Conversation display and trace history are held in the
browser session.

```bash
npm install
npm run dev
```

Install the Python dependencies listed under [Run it](#run-it) as well as the
Node dependencies. Start Next.js from the project root so the API can launch
Python against the included model and data files. Open the local URL printed
by Next.js (typically `http://localhost:3000`).

## Preference-aware execution

The agent loads the guest's predicted setup from
`data/guest_predicted_setups.csv` through `PreferenceTool` before executing
room-service actions. Preference knowledge is applied automatically when it
can resolve an otherwise incomplete request:

- A generic request such as _"order me something to eat"_ uses the guest's
  `dining_pref` value to select a compatible room-service meal.
- An explicit request such as _"order me coffee"_ remains explicit, while
  the guest's dining preference filters the available items for compatibility.
- If no preference is available and the request does not identify a food or
  drink category, the agent escalates to staff instead of guessing.
- Preference-backed orders still use the normal room-service availability
  checks and are escalated if no compatible item can be fulfilled.

The current mock menu supports `snacks`, `coffee`, and `breakfast`. Dietary
preferences currently represented in the prediction data include
`vegetarian`, `vegan`, and `non_vegetarian`.

## The safety guard: results

This is the "recognize abnormal input" piece, and it's evaluated the same
way as every other model in this series — a hand-labeled test set, not
just a few examples that happened to work:

```
Safety guard evaluation (23 hand-labeled examples)
------------------------------------------------------------
Overall accuracy: 100.0% (23/23)
False positives (normal input wrongly flagged): 0
False negatives (abnormal input missed): 0
```

Four categories, each handled differently because they need different
responses, not just a generic "rejected":

- **Emergency** (fire, gas, medical, intrusion) → bypasses all planning,
  escalates to staff immediately at URGENT priority. Deliberately biased
  toward false positives over false negatives here — an unnecessary staff
  trip costs far less than a missed real emergency.
- **Injection / out-of-authority** (unlocking doors, revealing other
  guests' data, "ignore your instructions") → refused outright and logged
  for staff review, not "interpreted charitably."
- **Gibberish** (keyboard mashing, empty input) → asked to rephrase, never
  forced through the classifier for a low-confidence guess.
- **Conflict** (e.g. "turn the lights on and off") → caught _after_
  planning, once sub-intents are known, and the guest is asked to clarify
  rather than the agent picking one arbitrarily.

**A real bug I found while building the gibberish detector:** the first
version checked "does this word contain a vowel," which sounds reasonable
but misses obvious cases — random keyboard mashing like "asdkjfh" still
has an 'a' in it. It also completely missed single-word gibberish like
"zzz" because the check only ran when there were 2+ words. I replaced it
with a proper implausible-word detector (checks for 4+ consecutive
consonants and repeated-single-character strings), which is what actually
catches both cases — worth including here rather than quietly fixing it,
since "the naive version looked fine until I specifically tested short
single-word input" is a more useful thing to be able to say than pretending
it worked the first time.

**Known, deliberate limitation:** the guard is keyword/pattern-based, not
ML-based, on purpose — for a safety-critical gate, being able to point at
the exact rule that fired matters more than squeezing out extra accuracy
from a black box. This means genuinely novel phrasings of an emergency
that don't match any pattern (e.g. an unusual way of describing a medical
issue) could slip through. A production system would pair this with a
human-in-the-loop review of near-miss confidence scores, not rely on the
pattern list alone.

## Why no real LLM-based planner

The planner in `planner.py` uses scene templates + the existing intent
classifier, not an LLM. This is an honest constraint, not a design
preference: an LLM-based planner needs API access, and this sandbox's
network allowlist doesn't include an LLM provider's endpoint. Here's the
shape of the swap-in if you have API access:

```python
import anthropic

client = anthropic.Anthropic()

def llm_plan(guest_request: str) -> list[dict]:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Break this hotel guest request into steps. For
each step, output {{"device": ..., "action": ...}}. Request: {guest_request}"""
        }],
    )
    # parse response.content into a list of step dicts
    ...
```

The tradeoff: an LLM planner generalizes far better to genuinely novel
multi-step requests than a fixed template list ever could, at the cost of
latency, cost per request, and losing the "I can point to the exact rule
that produced this plan" auditability that the current rule-based
approach has. For a safety-adjacent system controlling physical hardware,
that auditability tradeoff is worth naming explicitly, not glossed over.

## Project structure

```
hotel-concierge-agent/
├── agent.py              # orchestrator: guard -> plan -> check -> execute -> remember
├── planner.py            # scene templates + per-clause intent classification
├── safety_guard.py       # emergency / injection / gibberish / conflict detection
├── tools.py              # classifier, preference, device, room-service, and escalation tools
├── synonym_utils.py      # reused from voice-intent-classifier (needed to unpickle the model)
├── typo_utils.py         # reused from voice-intent-classifier
├── demo_scenarios.py     # guard evaluation + scripted end-to-end scenarios
├── dashboard.py          # live Streamlit chat interface
├── resources/
│   └── intent_model.joblib   # trained classifier, reused from voice-intent-classifier
├── data/
│   ├── guests.csv             # reused from guest-preference-engine
│   └── guest_predicted_setups.csv
└── output/
    ├── guard_evaluation.txt
    └── agent_run_log.txt
```

## Ideas to extend it

- Wire in the real LLM-based planner above once you have API access, and
  compare its plans against the template-based ones on genuinely novel
  multi-part requests.
- Apply the preference engine to additional executable actions, such as
  using a guest's lighting, room-temperature, TV, or wake-up preferences
  when a request leaves those details unspecified.
- Add a proper escalation dashboard for staff (currently just a Python
  list) — e.g. a second Streamlit page showing pending escalations by
  priority.
- Track a rolling risk score per guest session rather than treating every
  message independently — repeated injection attempts from the same
  guest_id is a stronger signal than one isolated message.
