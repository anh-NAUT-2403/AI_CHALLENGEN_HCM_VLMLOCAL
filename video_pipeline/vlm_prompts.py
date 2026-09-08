# video_pipeline/vlm_prompts.py

import json
from typing import Any, Dict, List

SYSTEM_PROMPT = """
You are a visual reasoning assistant for video retrieval.

Follow the task-specific prompt exactly.

For KIS:
- judge visual similarity to the original query
- be tolerant to small variations

For QA:
- answer the requested target directly from the visible clip
- do not require the entire query context to be verified
- focus on text, numbers, counts, objects, people, animals, colors, and attributes
- if evidence is insufficient, still return the most plausible best guess rather than null

For TRAKE:
- identify the requested events in temporal order
- focus on core visual anchors

Always return valid JSON only.
""".strip()


# ============================================================
# HELPERS
# ============================================================

def _original_query(
    analysis: Dict[str, Any]
) -> str:
    """
    Exact BTC/user query. VLM must use this, never expanded queries.
    """
    return str(
        analysis.get(
            "original_query",
            ""
        )
    ).strip()


def _qa_question(
    analysis: Dict[str, Any]
) -> str:
    """
    Analyzer's extracted QUESTION only.
    This is not used as a new semantic requirement; it isolates what QA must answer.
    """
    question = analysis.get("question")

    if question is None:
        return _original_query(analysis)

    question = str(question).strip()

    return (
        question
        if question
        else _original_query(analysis)
    )


def _qa_answer_type(
    analysis: Dict[str, Any]
) -> str:
    value = analysis.get("answer_type")

    if value is None:
        return "unknown"

    value = str(value).strip()

    return value or "unknown"


def _segment_duration(
    candidate: Dict[str, Any]
) -> float:
    start = _segment_start(candidate)
    end = _segment_end(candidate)

    return max(
        0.0,
        end - start,
    )


def _segment_start(
    candidate: Dict[str, Any]
) -> float:
    try:
        return float(
            candidate.get(
                "segment_start",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return 0.0


def _segment_end(
    candidate: Dict[str, Any]
) -> float:
    try:
        return float(
            candidate.get(
                "segment_end",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return 0.0


def _original_trake_events(
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    VLM uses original_description when main.py preserved it.
    Analyzer description/action remains retrieval-only.
    """
    output = []

    for index, event in enumerate(
        analysis.get(
            "events",
            []
        ),
        start=1,
    ):
        if not isinstance(
            event,
            dict,
        ):
            continue

        description = (
            event.get(
                "original_description"
            )
            or event.get(
                "description"
            )
            or event.get(
                "action"
            )
            or ""
        )

        output.append({
            "event_id": event.get(
                "event_id",
                index,
            ),
            "description": description,
        })

    return output


# ============================================================
# KIS
# ============================================================

def build_kis_prompt(
    analysis: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:

    original_query = _original_query(
        analysis
    )

    duration = _segment_duration(
        candidate
    )

    return f"""
TASK TYPE: KIS

ORIGINAL USER QUERY:
{original_query}

CANDIDATE CLIP:
Video ID: {candidate.get("video_id")}
Clip duration: {duration:.2f} seconds

IMPORTANT TIME RULE:
You are seeing a SHORT CUT CLIP.
Do NOT output the timestamp from the original full video.
best_relative_time MUST be between 0 and {duration:.2f}.
0 means the first frame of this supplied clip.

TASK:
Estimate how well this candidate visually matches the ORIGINAL USER QUERY.

Be tolerant.
Focus on the most important visual anchors:
- approximate people count
- important colors
- key objects
- main action
- distinctive arrangement

Do not reject due to camera angle, brief occlusion, lighting, blur, small pose
differences, or irrelevant background differences.

Use a continuous confidence:
- 0.85 - 1.00: very strong
- 0.65 - 0.84: strong
- 0.45 - 0.64: plausible
- 0.20 - 0.44: weak but meaningful
- 0.00 - 0.19: mostly unrelated

verified=true when the candidate is reasonably likely to contain the requested scene.

best_relative_time:
- choose the clearest matching moment INSIDE THIS SHORT CLIP
- MUST satisfy 0 <= best_relative_time <= {duration:.2f}
- never return an original-video timestamp

Judge ONLY the ORIGINAL USER QUERY.
Do NOT use Analyzer expansions as extra requirements.

Return ONLY valid JSON:
{{
  "verified": false,
  "confidence": 0.0,
  "best_relative_time": null,
  "reason": ""
}}
""".strip()


# ============================================================
# QA — ANSWER FIRST
# ============================================================

def build_qa_prompt(
    analysis: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:

    original_query = _original_query(
        analysis
    )

    question = _qa_question(
        analysis
    )

    answer_type = _qa_answer_type(
        analysis
    )

    duration = _segment_duration(
        candidate
    )

    return f"""
TASK TYPE: QA

QUESTION TO ANSWER:
{question}

EXPECTED ANSWER TYPE:
{answer_type}

ORIGINAL QUERY CONTEXT:
{original_query}

CANDIDATE CLIP:
Video ID: {candidate.get("video_id")}
Clip duration: {duration:.2f} seconds

YOUR ONLY JOB:
Look at THIS CLIP and answer the QUESTION TO ANSWER.

DO NOT score whether the whole original query is correct.
DO NOT require every event or contextual clue in ORIGINAL QUERY CONTEXT to be
visible in this clip.
DO NOT reject merely because an earlier/later action mentioned in the context
is absent.

The context is ONLY a locator hint telling you WHICH object/text/person/number
the question refers to.

Examples of the correct behavior:
- If the question asks for the NUMBER ON A SCALE:
  find the relevant scale in the clip and read its numeric display.
  Do NOT require a later shot of somebody holding another fish.
- If the question asks HOW MANY objects:
  count those requested objects in the clearest frame.
- If the question asks for a NAME/PLACE/TEXT:
  read the requested text from subtitles, signs, banners, labels or overlays.
- If the question asks WHAT OBJECT/ANIMAL/PERSON:
  identify that requested target directly.

TARGET-FIRST INSPECTION:
1. Identify exactly what the question wants as the answer.
2. Search the supplied clip for that target.
3. If the target is visible/readable/countable, return the answer immediately.
4. Ignore unrelated missing context.
5. If the target is unclear, partially hidden, blurry, or only weakly supported,
   still return your MOST PLAUSIBLE BEST GUESS.
6. NEVER return answer=null.

For numeric displays:
- inspect multiple frames because digits may become clearer briefly
- distinguish stable final reading from transient changing digits when possible
- copy only the requested number/unit
- do not reject a readable scale merely because another contextual event is absent

For text:
- inspect subtitles, labels, signs, banners, overlays and logos carefully.

For counting:
- use the clearest frame and count the requested class only.

confidence:
- confidence that this clip contains the requested answer target.
- NOT confidence that every statement in the whole original query is verified.

answer_confidence:
- confidence that the returned answer itself is correct.
- If answer is null, answer_confidence must be 0.0.

verified:
- true when a concrete answer is visually supported.
- false when the answer is only a weak best guess.

IMPORTANT TIME RULE:
best_relative_time is relative to THIS SHORT CLIP.
It MUST be between 0 and {duration:.2f}.
Never output an original-video timestamp.

ANSWER RULES:
- concise answer only
- maximum 100 characters
- prefer direct visual evidence
- if direct evidence is insufficient, make the single most plausible best guess
  from partial visual evidence and query context
- NEVER return null, empty string, "unknown", "N/A", or "cannot determine"
- when guessing, set verified=false and answer_confidence between 0.01 and 0.20

Return ONLY valid JSON:
{{
  "verified": false,
  "confidence": 0.0,
  "answer": "best guess",
  "answer_confidence": 0.0,
  "best_relative_time": null,
  "reason": ""
}}
""".strip()


# ============================================================
# TRAKE
# ============================================================

def build_trake_prompt(
    analysis: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:

    original_query = _original_query(
        analysis
    )

    events = _original_trake_events(
        analysis
    )

    expected_count = len(
        events
    )

    duration = _segment_duration(
        candidate
    )

    required_events_text = json.dumps(
        events,
        ensure_ascii=False,
        indent=2,
    )

    output_skeleton = [
        {
            "event_id": event.get(
                "event_id",
                index,
            ),
            "matched": False,
            "relative_time": None,
            "confidence": 0.0,
        }
        for index, event in enumerate(
            events,
            start=1,
        )
    ]

    skeleton_text = json.dumps(
        output_skeleton,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
TASK TYPE: TRAKE

ORIGINAL USER QUERY:
{original_query}

REQUIRED ORIGINAL EVENTS:
{required_events_text}

NUMBER OF REQUIRED EVENTS:
{expected_count}

CANDIDATE CLIP:
Video ID: {candidate.get("video_id")}
Clip duration: {duration:.2f} seconds

TASK:
Find the required events in temporal order.

Be tolerant and focus on the DISTINCTIVE CORE ANCHORS of each event.

Prioritize:

1. DISTINCTIVE ANIMALS / CHARACTERS
   - species/type
   - important body color
   - unusual markings
   - distinctive colored spots
   - nearby symbols or props

2. DISTINCTIVE OBJECTS / SYMBOLS
   - flags
   - dragons or dragon-like figures
   - poles / pillars / vertical bars
   - tools
   - containers
   - vehicles
   - signs
   - other explicitly requested objects

3. PEOPLE
   - presence of a person/group
   - interaction with an important object
   - relative position only when it helps distinguish the event

4. COLOR + OBJECT COMBINATIONS
   Examples of the type of evidence to prioritize:
   - a white animal with a red marking
   - an object beside a white flag
   - a dragon, a person, and a pillar together

   These are ONLY EXAMPLES.
   They are NOT extra requirements unless present in the ORIGINAL QUERY.

5. ACTION / STATE CHANGE
   - entering/leaving
   - touching
   - moving
   - placing
   - appearing beside another anchor
   - first occurrence / last occurrence when requested

6. TEMPORAL ORDER
   - E1 before E2, E2 before E3, and so on

TOLERANCE:
- Do not reject due to camera angle, blur, partial occlusion, small shade changes,
  or minor pose differences.
- Core animal/person/object/color/action combinations matter most.
- Use intermediate confidence for partial but meaningful evidence.
- matched=false should mean the CORE event is missing or strongly contradicted,
  not that one tiny detail is unclear.

OUTPUT RULES:
- Return EXACTLY {expected_count} event objects.
- NEVER omit an event.
- Preserve event order.
- If an event is not found:
    matched = false
    relative_time = null
    confidence = 0.0
- relative_time is measured from START of this supplied short clip.
- every relative_time MUST be between 0 and the clip duration.
- NEVER output original full-video timestamps.
- Overall confidence represents confidence in the full ordered sequence.
- verified can be true when all CORE events are reasonably supported in order;
  every tiny detail does not need to be perfect.

Judge ONLY the ORIGINAL USER QUERY and ORIGINAL EVENTS.
Do NOT use Analyzer expanded queries or inferred fields.

Required event output skeleton:
{skeleton_text}

Return ONLY valid JSON:
{{
  "verified": false,
  "confidence": 0.0,
  "events": {skeleton_text},
  "reason": ""
}}
""".strip()


# ============================================================
# DISPATCH
# ============================================================

def build_vlm_prompt(
    analysis: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:

    task = str(
        analysis.get(
            "task",
            ""
        )
    ).upper()

    if task == "KIS":
        return build_kis_prompt(
            analysis,
            candidate,
        )

    if task == "QA":
        return build_qa_prompt(
            analysis,
            candidate,
        )

    if task == "TRAKE":
        return build_trake_prompt(
            analysis,
            candidate,
        )

    raise ValueError(
        f"Unsupported VLM task: {task}"
    )
