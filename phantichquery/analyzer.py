
import os
import json
from typing import Dict, Any, List, Optional

from openai import OpenAI

from .prompts import (
    SYSTEM_PROMPT,
    build_task_detection_prompt,
    build_analysis_prompt
)


# ============================================================
# CONFIG
# ============================================================

# API key được đọc từ biến môi trường OPENAI_API_KEY hoặc điền khi chạy
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# TODO: Điền model bạn muốn sử dụng
MODEL_NAME = "gpt-5.6-luna"


# ============================================================
# CLIENT
# ============================================================

def get_client() -> OpenAI:
    """
    Tạo OpenAI client.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY đang trống. "
            "Hãy điền API key trong analyzer.py."
        )

    if not MODEL_NAME:
        raise ValueError(
            "MODEL_NAME đang trống. "
            "Hãy điền model trong analyzer.py."
        )

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# LOW LEVEL LLM CALL
# ============================================================

def call_llm(prompt: str) -> Dict[str, Any]:
    """
    Gọi LLM và parse JSON.

    Input:
        prompt: prompt đã hoàn chỉnh.

    Output:
        Python dictionary.
    """

    client = get_client()

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:

        raise ValueError(
            "LLM không trả về JSON hợp lệ.\n\n"
            f"Raw output:\n{text}"
        ) from e


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(
    raw_query: str,
    known_task: Optional[str] = None
) -> str:
    """
    Input:
        raw_query
        known_task (optional)

    Output:
        KIS / QA / TRAKE
    """

    if known_task:

        task = known_task.upper().strip()

        if task not in {
            "KIS",
            "QA",
            "TRAKE"
        }:
            raise ValueError(
                f"Task không hợp lệ: {known_task}"
            )

        return task

    prompt = build_task_detection_prompt(
        raw_query
    )

    result = call_llm(prompt)

    task = result.get(
        "task",
        ""
    ).upper()

    if task not in {
        "KIS",
        "QA",
        "TRAKE"
    }:
        raise ValueError(
            f"LLM trả task không hợp lệ: {task}"
        )

    return task


# ============================================================
# QUERY ANALYSIS
# ============================================================

def analyze_query(
    raw_query: str,
    task_type: str
) -> Dict[str, Any]:
    """
    Hàm LLM chính để phân tích query.

    Input:
        raw_query
        task_type

    Output:
        structured analysis dictionary
    """

    task_type = task_type.upper()

    prompt = build_analysis_prompt(
        raw_query=raw_query,
        task_type=task_type
    )

    return call_llm(prompt)


# ============================================================
# RETRIEVAL QUERY EXTRACTION
# ============================================================

def build_retrieval_queries(
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Lấy danh sách retrieval query từ analysis.

    Không gọi LLM.
    Không gọi FAISS.
    """

    return (
        analysis
        .get("retrieval", {})
        .get("expanded_queries", [])
    )


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(
    analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Lấy danh sách semantic events.

    Thứ tự phần tử trong list chính là temporal order.
    Không sử dụng field 'order'.

    Output:
        [
            {
                "event_id": 1,
                "description": "...",
                "action": "..."
            },
            ...
        ]
    """

    events = analysis.get(
        "events",
        []
    )

    normalized_events = []

    for i, event in enumerate(
        events,
        start=1
    ):

        # Trường hợp đúng schema
        if isinstance(event, dict):

            normalized_events.append(
                {
                    "event_id": event.get(
                        "event_id",
                        i
                    ),
                    "description": event.get(
                        "description",
                        ""
                    ),
                    "action": event.get(
                        "action",
                        ""
                    )
                }
            )

        # Fallback:
        # nếu LLM vẫn lỡ trả string thì không crash
        elif isinstance(event, str):

            normalized_events.append(
                {
                    "event_id": i,
                    "description": event,
                    "action": ""
                }
            )

        else:

            raise ValueError(
                f"Event {i} có format không hợp lệ: {event}"
            )

    return normalized_events


# ============================================================
# VALIDATION
# ============================================================

def validate_analysis(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Kiểm tra output trước khi gửi sang retrieval.
    """

    required_fields = [
        "task",
        "original_query",
        "retrieval",
        "entities",
        "attributes",
        "actions",
        "scenes",
        "events"
    ]

    for field in required_fields:

        if field not in analysis:

            raise ValueError(
                f"Thiếu field bắt buộc: {field}"
            )

    task = analysis["task"].upper()

    if task not in {
        "KIS",
        "QA",
        "TRAKE"
    }:

        raise ValueError(
            f"Task không hợp lệ: {task}"
        )

    retrieval = analysis["retrieval"]

    if not retrieval.get("main_query"):

        raise ValueError(
            "Thiếu retrieval.main_query"
        )

    queries = retrieval.get(
        "expanded_queries",
        []
    )

    if not queries:

        raise ValueError(
            "expanded_queries đang rỗng"
        )

    for i, query in enumerate(queries):

        if not query.get("text"):

            raise ValueError(
                f"Retrieval query {i} thiếu text"
            )

        if "weight" not in query:

            raise ValueError(
                f"Retrieval query {i} thiếu weight"
            )

        weight = query["weight"]

        if not isinstance(
            weight,
            (int, float)
        ):

            raise ValueError(
                f"Weight của query {i} không phải số"
            )

    if task == "QA":

        if not analysis.get("question"):

            raise ValueError(
                "Task QA nhưng không có question"
            )

    if task == "TRAKE":

        events = analysis.get(
            "events",
            []
        )

        if len(events) < 2:

            raise ValueError(
                "TRAKE phải có ít nhất 2 events"
            )

    return analysis


# ============================================================
# HIGH LEVEL INTERFACE
# ============================================================

def analyze(
    raw_query: str,
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hàm chính mà các module khác nên gọi.

    raw query
        ↓
    detect task
        ↓
    LLM analysis
        ↓
    validation
        ↓
    structured JSON
    """

    task = detect_task(
        raw_query=raw_query,
        known_task=task_type
    )

    analysis = analyze_query(
        raw_query=raw_query,
        task_type=task
    )

    return validate_analysis(
        analysis
    )
