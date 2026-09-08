import csv
import os

# Qwen video backend: avoid broken torchvision/torchcodec fallback on Colab.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
import re
import shutil
import zipfile
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CLIP_DEVICE,
    CLIP_MODEL_NAME,
    DEBUG_RETRIEVAL,
    DEBUG_VIDEO_PIPELINE,
    FAISS_INDEX_PATH,
    FUSION_BEST_WEIGHT,
    FUSION_COVERAGE_WEIGHT,
    FUSION_SECOND_WEIGHT,
    METADATA_PATH,
    OBJECT_BOOST_WEIGHT,
    OBJECT_DETECTION_THRESHOLD,
    OBJECT_SEMANTIC_THRESHOLD,
    OBJECT_ZIP_PATH,
    QUERY_PACKAGE_PATH,
    RETRIEVAL_TOP_K,
    RETRIEVAL_TOP_N,
    SEGMENT_MAX_DURATION,
    SEGMENT_MIN_DURATION,
    SEGMENT_PADDING_AFTER,
    SEGMENT_PADDING_BEFORE,
    SEGMENT_ROOT,
    SUBMISSION_FRAME_MODE,
    SUBMISSION_KIS_MAX_LINES,
    SUBMISSION_QA_MAX_LINES,
    SUBMISSION_QA_REQUIRE_VERIFIED,
    SUBMISSION_STRICT,
    SUBMISSION_TRAKE_ALLOW_RETRIEVAL_FALLBACK,
    SUBMISSION_TRAKE_MAX_LINES,
    SUBMISSION_TRAKE_REQUIRE_VERIFIED,
    SUBMISSION_WORK_ROOT,
    SUBMISSION_ZIP_PATH,
    TEMPORAL_MAX_CLUSTER_DURATION,
    TEMPORAL_MAX_GAP_SECONDS,
    TRAKE_ALIGNMENT_BEAM_SIZE,
    TRAKE_ALIGNMENT_TOP_N,
    TRAKE_EVENT_TOP_K,
    TRAKE_GAP_PENALTY_WEIGHT,
    TRAKE_HITS_PER_EVENT_PER_VIDEO,
    TRAKE_MAX_EVENT_GAP_SECONDS,
    TRAKE_MAX_TOTAL_SPAN_SECONDS,
    USE_OBJECT,
    USE_VLM,
    VIDEO_CACHE_ROOT,
    VIDEOS_ROOT,
    VLM_DEVICE_MAP,
    VLM_DTYPE,
    VLM_FPS_KIS,
    VLM_FPS_QA,
    VLM_FPS_TRAKE,
    VLM_MATCH_THRESHOLD,
    VLM_MAX_NEW_TOKENS,
    VLM_MINIMUM_FACTOR,
    VLM_MODEL_NAME,
    VLM_POSITIVE_BOOST,
    VLM_REJECT_PENALTY,
    VLM_VERIFY_TOP_N,
)
from phantichquery.analyzer import analyze
from retrieval.engine import RetrievalEngine
from retrieval.trake_aligner import TrakeAligner
from retrieval.trake_retriever import TrakeEventRetriever
from video_pipeline.engine import VideoPipelineEngine

# ============================================================
# GLOBAL ENGINES
# ============================================================

_RETRIEVAL_ENGINE = None
_VIDEO_PIPELINE_ENGINE = None
_FRAME_RESOLVER = None
_TRAKE_RETRIEVER = None
_TRAKE_ALIGNER = None

# QA is answer-first: use the answer when the VLM can actually read/see it.
# This threshold is deliberately permissive; raise it after calibration.
QA_MIN_ANSWER_CONFIDENCE = 0.0

# Keep Colab disk usage bounded during a 24-query run.
CLEAN_CACHE_AFTER_EACH_QUERY = True

# TRAKE fallback: if strict alignment returns zero videos, retry with looser timing.
TRAKE_RELAXED_FALLBACK_ENABLED = True
TRAKE_RELAXED_BEAM_SIZE = 100
TRAKE_RELAXED_HITS_PER_EVENT_PER_VIDEO = 50
TRAKE_RELAXED_MAX_EVENT_GAP_SECONDS = 900.0
TRAKE_RELAXED_MAX_TOTAL_SPAN_SECONDS = 1800.0
TRAKE_RELAXED_GAP_PENALTY_WEIGHT = 0.0

# Last-resort TRAKE fallback.
# Goal: never abort the whole package merely because TRAKE alignment found 0 videos.
TRAKE_FORCED_BEST_GUESS_ENABLED = True
TRAKE_FORCED_FRAME_STEP = 1


# ============================================================
# NATURAL SORT
# ============================================================

def _natural_sort_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


# ============================================================
# TASK FROM FILE NAME
# ============================================================

def infer_task_from_filename(filename: str) -> str:
    """
    BTC quy định:
        xxx-kis.txt
        xxx-qa.txt
        xxx-trake.txt
    Hậu tố filename là nguồn xác định task chính thức.
    """
    stem = Path(filename).stem.lower()
    match = re.search(r"-(kis|qa|trake)$", stem)

    if not match:
        raise ValueError(f"Không xác định được task từ filename: {filename}")

    return match.group(1).upper()


# ============================================================
# LOAD QUERY PACKAGE
# ============================================================

def load_query_package(source_path: str) -> List[Dict[str, Any]]:
    """
    Hỗ trợ:
        1. file ZIP BTC
        2. folder chứa *.txt
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Không tìm thấy query package: {source_path}")

    query_items = []

    # CASE 1 - ZIP
    if zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path, "r") as zf:
            members = [
                name for name in zf.namelist()
                if not name.endswith("/") and name.lower().endswith(".txt")
            ]

            for member in members:
                filename = os.path.basename(member)
                raw = zf.read(member)
                text = raw.decode("utf-8-sig").strip()

                query_items.append({
                    "query_name": Path(filename).stem,
                    "filename": filename,
                    "source_member": member,
                    "task": infer_task_from_filename(filename),
                    "text": text,
                })

    # CASE 2 - DIRECTORY
    elif os.path.isdir(source_path):
        txt_files = []
        for root, _, files in os.walk(source_path):
            for filename in files:
                if filename.lower().endswith(".txt"):
                    txt_files.append(os.path.join(root, filename))

        for path in txt_files:
            filename = os.path.basename(path)
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read().strip()

            query_items.append({
                "query_name": Path(filename).stem,
                "filename": filename,
                "source_member": path,
                "task": infer_task_from_filename(filename),
                "text": text,
            })
    else:
        raise ValueError("QUERY_PACKAGE_PATH phải là ZIP hoặc directory.")

    if not query_items:
        raise RuntimeError("Không tìm thấy file query .txt.")

    # DUPLICATE CHECK
    names = [item["query_name"] for item in query_items]
    if len(names) != len(set(names)):
        raise RuntimeError("Có query filename bị trùng.")

    query_items.sort(key=lambda x: _natural_sort_key(x["query_name"]))

    # INFO
    task_counts = {"KIS": 0, "QA": 0, "TRAKE": 0}
    for item in query_items:
        task_counts[item["task"]] += 1

    print("\n==========================================")
    print("QUERY PACKAGE LOADED")
    print("==========================================")
    print(f"Total : {len(query_items)}")
    print(f"KIS   : {task_counts['KIS']}")
    print(f"QA    : {task_counts['QA']}")
    print(f"TRAKE : {task_counts['TRAKE']}")
    print("==========================================\n")

    return query_items


# ============================================================
# TRAKE EVENT PARSER
# ============================================================

def extract_trake_events(text: str) -> List[Dict[str, Any]]:
    events = []
    current_label = None
    current_parts = []

    def flush():
        nonlocal current_label, current_parts
        if current_label is None:
            return

        description = " ".join(part.strip() for part in current_parts if part.strip()).strip()
        events.append({
            "source_label": current_label,
            "description": description,
        })
        current_label = None
        current_parts = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        match = re.match(r"^(E\d+)\s*:\s*(.*)$", stripped, flags=re.IGNORECASE)
        if match:
            flush()
            current_label = match.group(1).upper()
            current_parts = [match.group(2)]
        elif current_label is not None:
            current_parts.append(stripped)

    flush()

    normalized = []
    for order, event in enumerate(events, start=1):
        normalized.append({
            "event_id": order,
            "order": order,
            "source_label": event["source_label"],
            "description": event["description"],
            "action": event["description"],
        })

    return normalized


# ============================================================
# PREPARE ANALYSIS
# ============================================================

def prepare_analysis(query_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzer is used to improve RETRIEVAL recall.

    VLM verification/answering must still see the exact original BTC query.
    For TRAKE:
      - original event wording is preserved for the VLM;
      - Analyzer event wording can be used by CLIP event retrieval.
    """
    raw_query = query_item["text"]
    official_task = query_item["task"]

    analysis = analyze(raw_query)
    analyzer_task = str(analysis.get("task", "")).upper()

    analysis["task_detected_by_analyzer"] = analyzer_task
    analysis["task"] = official_task
    analysis["query_name"] = query_item["query_name"]

    # CRITICAL:
    # Never allow an Analyzer rewrite to replace the exact BTC query for VLM.
    analysis["original_query"] = raw_query

    if analyzer_task and analyzer_task != official_task:
        print("[WARNING] Analyzer task mismatch:")
        print(f"    Analyzer : {analyzer_task}")
        print(f"    Filename : {official_task}")
        print("    -> sử dụng task từ filename BTC.")

    if official_task == "TRAKE":
        source_events = extract_trake_events(raw_query)
        analyzer_events = analysis.get("events", [])

        if source_events:
            merged_events = []

            for index, source_event in enumerate(source_events, start=1):
                analyzer_event = (
                    analyzer_events[index - 1]
                    if index - 1 < len(analyzer_events)
                    and isinstance(analyzer_events[index - 1], dict)
                    else {}
                )

                original_description = source_event.get("description", "")

                # Retrieval-oriented text: prefer Analyzer's event description/action,
                # because it is usually more CLIP-friendly (often English).
                retrieval_description = (
                    analyzer_event.get("description")
                    or analyzer_event.get("action")
                    or original_description
                )

                retrieval_action = (
                    analyzer_event.get("action")
                    or analyzer_event.get("description")
                    or original_description
                )

                merged_events.append({
                    "event_id": index,
                    "order": index,
                    "source_label": source_event.get("source_label", f"E{index}"),

                    # Exact BTC wording for VLM.
                    "original_description": original_description,

                    # Retrieval wording for TrakeEventRetriever / CLIP.
                    "description": retrieval_description,
                    "action": retrieval_action,
                })

            analysis["events"] = merged_events
            analysis["expected_event_count"] = len(merged_events)

        else:
            analysis["expected_event_count"] = len(
                analysis.get("events", [])
            )

    return analysis


# ============================================================
# RETRIEVAL ENGINE
# ============================================================

def get_retrieval_engine():
    global _RETRIEVAL_ENGINE
    if _RETRIEVAL_ENGINE is None:
        object_zip_path = OBJECT_ZIP_PATH if USE_OBJECT else None
        _RETRIEVAL_ENGINE = RetrievalEngine(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            object_zip_path=object_zip_path,
            top_k=RETRIEVAL_TOP_K,
            top_n=RETRIEVAL_TOP_N,
            max_gap_seconds=TEMPORAL_MAX_GAP_SECONDS,
            max_cluster_duration=TEMPORAL_MAX_CLUSTER_DURATION,
            best_weight=FUSION_BEST_WEIGHT,
            second_weight=FUSION_SECOND_WEIGHT,
            coverage_weight=FUSION_COVERAGE_WEIGHT,
            object_detection_threshold=OBJECT_DETECTION_THRESHOLD,
            object_semantic_threshold=OBJECT_SEMANTIC_THRESHOLD,
            object_boost_weight=OBJECT_BOOST_WEIGHT,
            clip_model_name=CLIP_MODEL_NAME,
            device=CLIP_DEVICE,
        )
    return _RETRIEVAL_ENGINE


# ============================================================
# TRAKE MODULES
# ============================================================

def get_trake_modules():
    global _TRAKE_RETRIEVER
    global _TRAKE_ALIGNER

    retrieval_engine = get_retrieval_engine()

    if _TRAKE_RETRIEVER is None:
        _TRAKE_RETRIEVER = TrakeEventRetriever(
            store=retrieval_engine.store,
            encoder=retrieval_engine.encoder,
            top_k=TRAKE_EVENT_TOP_K,
        )

    if _TRAKE_ALIGNER is None:
        _TRAKE_ALIGNER = TrakeAligner(
            top_n=TRAKE_ALIGNMENT_TOP_N,
            beam_size=TRAKE_ALIGNMENT_BEAM_SIZE,
            hits_per_event_per_video=TRAKE_HITS_PER_EVENT_PER_VIDEO,
            max_event_gap_seconds=TRAKE_MAX_EVENT_GAP_SECONDS,
            max_total_span_seconds=TRAKE_MAX_TOTAL_SPAN_SECONDS,
            gap_penalty_weight=TRAKE_GAP_PENALTY_WEIGHT,
        )

    return _TRAKE_RETRIEVER, _TRAKE_ALIGNER


# ============================================================
# VIDEO ENGINE
# ============================================================

def get_video_pipeline_engine():
    global _VIDEO_PIPELINE_ENGINE
    if _VIDEO_PIPELINE_ENGINE is None:
        _VIDEO_PIPELINE_ENGINE = VideoPipelineEngine(
            videos_root=VIDEOS_ROOT,
            cache_root=VIDEO_CACHE_ROOT,
            segment_root=SEGMENT_ROOT,
            padding_before=SEGMENT_PADDING_BEFORE,
            padding_after=SEGMENT_PADDING_AFTER,
            min_segment_duration=SEGMENT_MIN_DURATION,
            max_segment_duration=SEGMENT_MAX_DURATION,
            enable_vlm=USE_VLM,
            vlm_model_name=VLM_MODEL_NAME,
            vlm_device_map=VLM_DEVICE_MAP,
            vlm_dtype=VLM_DTYPE,
            vlm_fps_kis=VLM_FPS_KIS,
            vlm_fps_qa=VLM_FPS_QA,
            vlm_fps_trake=VLM_FPS_TRAKE,
            vlm_max_new_tokens=VLM_MAX_NEW_TOKENS,
            vlm_positive_boost=VLM_POSITIVE_BOOST,
            vlm_reject_penalty=VLM_REJECT_PENALTY,
            vlm_minimum_factor=VLM_MINIMUM_FACTOR,
            verify_top_n=VLM_VERIFY_TOP_N,
        )
    return _VIDEO_PIPELINE_ENGINE


# ============================================================
# FRAME RESOLVER
# ============================================================

class FrameResolver:
    def __init__(self, metadata: List[Dict[str, Any]]):
        grouped = {}
        for item in metadata:
            video_id = item.get("video_id")
            if not video_id:
                continue
            grouped.setdefault(video_id, []).append((
                float(item["pts_time"]),
                int(item["frame_id"]),
                float(item.get("fps", 0.0)),
            ))

        self.data = {}
        for video_id, items in grouped.items():
            items.sort(key=lambda x: x[0])
            self.data[video_id] = {
                "times": [x[0] for x in items],
                "frames": [x[1] for x in items],
                "fps": items[0][2] if items else 0.0,
            }

    def nearest(
        self,
        video_id: str,
        time_sec: float,
        fallback_frame: Optional[int] = None,
    ) -> Optional[int]:
        data = self.data.get(video_id)
        if not data:
            return fallback_frame

        times = data["times"]
        frames = data["frames"]
        if not times:
            return fallback_frame

        target = float(time_sec)
        idx = bisect_left(times, target)
        candidate_indices = []

        if idx < len(times):
            candidate_indices.append(idx)
        if idx > 0:
            candidate_indices.append(idx - 1)

        if not candidate_indices:
            return fallback_frame

        best_idx = min(candidate_indices, key=lambda i: abs(times[i] - target))
        return int(frames[best_idx])

    def nearest_sequence(
        self,
        video_id: str,
        event_times: List[float],
    ) -> Optional[List[int]]:
        data = self.data.get(video_id)
        if not data:
            return None

        times = data["times"]
        frames = data["frames"]
        output = []
        minimum_index = 0

        for event_time in event_times:
            if minimum_index >= len(times):
                return None

            target = float(event_time)
            idx = bisect_left(times, target, lo=minimum_index)
            candidates = []

            if idx < len(times):
                candidates.append(idx)
            if idx - 1 >= minimum_index:
                candidates.append(idx - 1)

            if not candidates:
                return None

            best_idx = min(candidates, key=lambda i: abs(times[i] - target))
            output.append(int(frames[best_idx]))
            minimum_index = best_idx + 1

        return output


def get_frame_resolver():
    global _FRAME_RESOLVER
    if _FRAME_RESOLVER is None:
        retrieval_engine = get_retrieval_engine()
        _FRAME_RESOLVER = FrameResolver(retrieval_engine.store.metadata)
    return _FRAME_RESOLVER


# ============================================================
# VIDEO ID CLEANUP & TIME RESOLUTION
# ============================================================

def clean_video_id(video_id: str) -> str:
    video_id = os.path.basename(str(video_id))
    if video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]
    return video_id


def get_candidate_time(candidate: Dict[str, Any]) -> Optional[float]:
    """
    Return an ABSOLUTE time in the original video.

    IMPORTANT:
    For KIS/QA, Retrieval already gives a reliable candidate_time around the
    matched keyframe. Qwen's numeric best_relative_time can occasionally be
    interpreted inconsistently (relative vs absolute), especially when the
    prompt also exposes original segment timestamps.

    Therefore:
      1) Prefer Retrieval candidate_time.
      2) Only use vlm_best_time as a defensive fallback.
      3) If vlm_best_time looks relative to the segment, convert it.
      4) If it looks absolute and lies inside the segment, accept it.
      5) Otherwise reject it instead of producing a wildly wrong frame.
    """
    candidate_time = candidate.get("candidate_time")

    if candidate_time is not None:
        try:
            return float(candidate_time)
        except (TypeError, ValueError):
            pass

    vlm_time = candidate.get("vlm_best_time")

    if vlm_time is None:
        return None

    try:
        vlm_time = float(vlm_time)
    except (TypeError, ValueError):
        return None

    try:
        segment_start = float(candidate.get("segment_start"))
        segment_end = float(candidate.get("segment_end"))
    except (TypeError, ValueError):
        return None

    if segment_end < segment_start:
        return None

    duration = segment_end - segment_start
    tolerance = 1.5

    # Looks like a relative time inside the cut segment.
    if -tolerance <= vlm_time <= duration + tolerance:
        relative = min(max(vlm_time, 0.0), duration)
        return segment_start + relative

    # Looks like an absolute source-video time.
    if (
        segment_start - tolerance
        <= vlm_time
        <= segment_end + tolerance
    ):
        return min(
            max(vlm_time, segment_start),
            segment_end,
        )

    # Ambiguous / impossible VLM timestamp -> do not trust it.
    return None


def resolve_candidate_frame(candidate: Dict[str, Any]) -> Optional[int]:
    fallback = candidate.get("candidate_frame")
    if fallback is not None:
        try:
            fallback = int(fallback)
        except (TypeError, ValueError):
            fallback = None

    if SUBMISSION_FRAME_MODE == "candidate":
        return fallback

    if SUBMISSION_FRAME_MODE == "nearest_keyframe":
        time_sec = get_candidate_time(candidate)
        if time_sec is None:
            return fallback

        resolver = get_frame_resolver()
        return resolver.nearest(
            video_id=candidate["video_id"],
            time_sec=time_sec,
            fallback_frame=fallback,
        )

    raise ValueError(f"Unknown SUBMISSION_FRAME_MODE: {SUBMISSION_FRAME_MODE}")


# ============================================================
# FINAL ORDERING
# ============================================================

def _reorder_after_vlm(
    retrieval_candidates: List[Dict[str, Any]],
    video_candidates: List[Dict[str, Any]],
    verify_top_n: int,
) -> List[Dict[str, Any]]:
    """
    Only candidates actually sent to VLM are reranked against each other.

    Final layout:
        reranked VLM Top-N
        +
        untouched Retrieval remainder

    This avoids unchecked Retrieval candidates jumping above checked candidates
    merely because VLM penalized the checked block.
    """
    if not video_candidates:
        return []

    verify_n = min(
        max(int(verify_top_n), 0),
        len(retrieval_candidates),
    )

    checked_video_ids = {
        str(c.get("video_id"))
        for c in retrieval_candidates[:verify_n]
    }

    # video_candidates is already in VideoPipelineEngine's VLM-aware order.
    checked_block = [
        c
        for c in video_candidates
        if str(c.get("video_id")) in checked_video_ids
    ]

    updated_by_video = {
        str(c.get("video_id")): c
        for c in video_candidates
        if c.get("video_id") is not None
    }

    remainder = []

    for original in retrieval_candidates[verify_n:]:
        video_id = str(original.get("video_id"))

        remainder.append(
            updated_by_video.get(
                video_id,
                original,
            )
        )

    final_candidates = checked_block + remainder

    # Defensive de-duplication by video_id.
    deduped = []
    seen = set()

    for candidate in final_candidates:
        video_id = str(candidate.get("video_id"))

        if video_id in seen:
            continue

        seen.add(video_id)
        deduped.append(candidate)

    for rank, candidate in enumerate(deduped, start=1):
        candidate["rank"] = rank

    return deduped


# ============================================================
# CACHE CLEANUP
# ============================================================

def cleanup_query_cache():
    """
    Delete only temporary extracted videos and temporary VLM segments.

    Never touches:
      - Hugging Face model cache
      - source Videos_*.zip files on Drive
      - submission CSV files
    """
    paths = [
        os.path.join(VIDEO_CACHE_ROOT, "videos"),
        SEGMENT_ROOT,
    ]

    for path in paths:
        if not path:
            continue

        try:
            shutil.rmtree(
                path,
                ignore_errors=True,
            )
            os.makedirs(
                path,
                exist_ok=True,
            )
        except Exception as error:
            print(
                f"[WARNING] Không cleanup được cache {path}: {error}"
            )

    print("[CACHE] video + segment cache cleared.")


# ============================================================
# RELAXED TRAKE FALLBACK
# ============================================================

def run_relaxed_trake_alignment(
    event_results,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retry TRAKE alignment with wider temporal constraints when strict
    alignment returns zero candidate videos.
    """
    if not TRAKE_RELAXED_FALLBACK_ENABLED:
        return []

    print("\n==========================================")
    print("TRAKE RELAXED FALLBACK")
    print("==========================================")
    print("Strict alignment returned 0 candidates.")
    print("Retrying with relaxed temporal constraints...")

    relaxed_aligner = TrakeAligner(
        top_n=TRAKE_ALIGNMENT_TOP_N,
        beam_size=TRAKE_RELAXED_BEAM_SIZE,
        hits_per_event_per_video=TRAKE_RELAXED_HITS_PER_EVENT_PER_VIDEO,
        max_event_gap_seconds=TRAKE_RELAXED_MAX_EVENT_GAP_SECONDS,
        max_total_span_seconds=TRAKE_RELAXED_MAX_TOTAL_SPAN_SECONDS,
        gap_penalty_weight=TRAKE_RELAXED_GAP_PENALTY_WEIGHT,
    )

    candidates = relaxed_aligner.align(
        event_results=event_results,
        debug=debug,
    )

    print("Relaxed TRAKE candidates:", len(candidates))
    return candidates


# ============================================================
# FORCED TRAKE BEST-GUESS FALLBACK
# ============================================================

def _trake_hit_score(hit: Dict[str, Any]) -> float:
    for key in (
        "score",
        "final_score",
        "semantic_score",
        "retrieval_score",
        "weighted_score",
    ):
        value = hit.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return 0.0


def _trake_hit_time(hit: Dict[str, Any]) -> Optional[float]:
    for key in (
        "time",
        "pts_time",
        "candidate_time",
        "timestamp",
    ):
        value = hit.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    frame = hit.get("frame_id")

    if frame is not None:
        try:
            # Most AIC metadata is 25/30 fps, but time should normally already exist.
            # This is only an emergency ordering fallback.
            return float(frame)
        except (TypeError, ValueError):
            pass

    return None


def _find_hit_dicts(obj: Any) -> List[Dict[str, Any]]:
    """
    Recursively find hit-like dictionaries inside one event result.

    This deliberately supports several possible TrakeEventRetriever output
    layouts so the fallback does not depend on one exact container key.
    """
    found = []

    if isinstance(obj, dict):
        if (
            obj.get("video_id") is not None
            and (
                obj.get("frame_id") is not None
                or obj.get("time") is not None
                or obj.get("pts_time") is not None
                or obj.get("candidate_time") is not None
            )
        ):
            found.append(obj)

        for value in obj.values():
            found.extend(
                _find_hit_dicts(value)
            )

    elif isinstance(obj, list):
        for item in obj:
            found.extend(
                _find_hit_dicts(item)
            )

    return found


def _event_hit_groups(
    event_results: Any,
) -> List[List[Dict[str, Any]]]:
    """
    Normalize event_results into:
        [
            [hits for E1],
            [hits for E2],
            ...
        ]
    """
    if isinstance(event_results, list):
        raw_events = event_results

    elif isinstance(event_results, dict):
        for key in (
            "events",
            "event_results",
            "results",
        ):
            value = event_results.get(key)

            if isinstance(value, list):
                raw_events = value
                break
        else:
            raw_events = [
                event_results
            ]

    else:
        raw_events = []

    groups = []

    for event_result in raw_events:
        hits = _find_hit_dicts(
            event_result
        )

        # De-duplicate obvious repeated recursive matches.
        unique = []
        seen = set()

        for hit in hits:
            key = (
                str(hit.get("video_id")),
                str(hit.get("frame_id")),
                str(
                    hit.get(
                        "time",
                        hit.get(
                            "pts_time",
                            hit.get(
                                "candidate_time"
                            )
                        )
                    )
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(hit)

        unique.sort(
            key=_trake_hit_score,
            reverse=True,
        )

        groups.append(unique)

    return groups


def build_forced_trake_candidates(
    event_results: Any,
    analysis: Dict[str, Any],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Last-resort TRAKE candidate constructor.

    Strategy:
      1) Find all per-event retrieval hits.
      2) Prefer one video that appears in the largest number of events.
      3) For each event, take that video's strongest hit when available.
      4) If an event has no hit in that video, reuse the nearest available
         information and enforce strictly increasing frames.
      5) Return one candidate with aligned_events.

    This is intentionally a BEST-GUESS fallback, not a semantic guarantee.
    It exists so one TRAKE query cannot abort an otherwise valid submission.
    """
    if not TRAKE_FORCED_BEST_GUESS_ENABLED:
        return []

    expected_count = int(
        analysis.get(
            "expected_event_count",
            len(
                analysis.get(
                    "events",
                    []
                )
            ),
        )
    )

    if expected_count <= 0:
        return []

    groups = _event_hit_groups(
        event_results
    )

    # Ensure number of groups matches expected events as closely as possible.
    if len(groups) < expected_count:
        groups.extend(
            [[] for _ in range(
                expected_count - len(groups)
            )]
        )

    groups = groups[:expected_count]

    # --------------------------------------------------------
    # Score videos by event coverage + retrieval quality
    # --------------------------------------------------------
    video_stats = {}

    for event_index, hits in enumerate(groups):
        best_per_video = {}

        for hit in hits:
            video_id = str(
                hit.get(
                    "video_id",
                    ""
                )
            ).strip()

            if not video_id:
                continue

            if (
                video_id not in best_per_video
                or _trake_hit_score(hit)
                > _trake_hit_score(
                    best_per_video[video_id]
                )
            ):
                best_per_video[video_id] = hit

        for video_id, hit in best_per_video.items():
            stat = video_stats.setdefault(
                video_id,
                {
                    "coverage": 0,
                    "score": 0.0,
                }
            )

            stat["coverage"] += 1
            stat["score"] += _trake_hit_score(
                hit
            )

    if not video_stats:
        print(
            "[TRAKE FORCED FALLBACK] "
            "Không tìm được hit nào để đoán."
        )
        return []

    best_video_id = max(
        video_stats,
        key=lambda vid: (
            video_stats[vid]["coverage"],
            video_stats[vid]["score"],
        ),
    )

    # --------------------------------------------------------
    # Pick one hit per event for the chosen video
    # --------------------------------------------------------
    chosen_hits = []

    for hits in groups:
        same_video_hits = [
            hit
            for hit in hits
            if str(
                hit.get(
                    "video_id",
                    ""
                )
            ) == best_video_id
        ]

        same_video_hits.sort(
            key=_trake_hit_score,
            reverse=True,
        )

        chosen_hits.append(
            same_video_hits[0]
            if same_video_hits
            else None
        )

    # Use any known frame as anchor.
    known_frames = []

    for hit in chosen_hits:
        if hit is None:
            continue

        try:
            known_frames.append(
                int(hit.get("frame_id"))
            )
        except (TypeError, ValueError):
            pass

    if not known_frames:
        # Fall back to best hit from any event for the chosen video.
        all_video_hits = []

        for hits in groups:
            all_video_hits.extend(
                [
                    h
                    for h in hits
                    if str(
                        h.get(
                            "video_id",
                            ""
                        )
                    ) == best_video_id
                ]
            )

        if not all_video_hits:
            return []

        all_video_hits.sort(
            key=_trake_hit_score,
            reverse=True,
        )

        anchor_hit = all_video_hits[0]

        try:
            anchor_frame = int(
                anchor_hit.get(
                    "frame_id",
                    0
                )
            )
        except (TypeError, ValueError):
            anchor_frame = 0

    else:
        anchor_frame = min(
            known_frames
        )

    # --------------------------------------------------------
    # Build strictly increasing best-effort sequence
    # --------------------------------------------------------
    aligned_events = []
    previous_frame = None

    for event_index in range(expected_count):
        hit = chosen_hits[event_index]

        frame_id = None
        event_time = None
        event_score = 0.0

        if hit is not None:
            try:
                frame_id = int(
                    hit.get(
                        "frame_id"
                    )
                )
            except (TypeError, ValueError):
                frame_id = None

            event_time = _trake_hit_time(
                hit
            )

            event_score = _trake_hit_score(
                hit
            )

        if frame_id is None:
            if previous_frame is None:
                frame_id = (
                    anchor_frame
                    + event_index * TRAKE_FORCED_FRAME_STEP
                )
            else:
                frame_id = (
                    previous_frame
                    + TRAKE_FORCED_FRAME_STEP
                )

        if (
            previous_frame is not None
            and frame_id <= previous_frame
        ):
            frame_id = (
                previous_frame
                + TRAKE_FORCED_FRAME_STEP
            )

        if event_time is None:
            # Emergency ordering value; FrameResolver/VLM may improve it later.
            event_time = float(
                frame_id
            )

        aligned_events.append({
            "event_id": event_index + 1,
            "frame_id": int(frame_id),
            "time": float(event_time),
            "score": float(event_score),
            "forced": True,
        })

        previous_frame = int(
            frame_id
        )

    candidate_frame = int(
        aligned_events[0]["frame_id"]
    )

    candidate_time = float(
        aligned_events[0]["time"]
    )

    stats = video_stats[
        best_video_id
    ]

    forced_candidate = {
        "video_id": best_video_id,
        "candidate_frame": candidate_frame,
        "candidate_time": candidate_time,
        "frame_id": candidate_frame,
        "time": candidate_time,
        "aligned_events": aligned_events,

        # Keep all common retrieval score aliases populated.
        "retrieval_score": float(
            stats["score"]
        ),
        "final_score": float(
            stats["score"]
        ),
        "semantic_score": float(
            stats["score"]
        ),

        "trake_forced_best_guess": True,
        "trake_event_coverage": int(
            stats["coverage"]
        ),
    }

    print("\n==========================================")
    print("TRAKE FORCED BEST-GUESS")
    print("==========================================")
    print(
        "Video:",
        best_video_id,
    )
    print(
        "Event coverage:",
        f"{stats['coverage']}/{expected_count}",
    )
    print(
        "Frames:",
        [
            e["frame_id"]
            for e in aligned_events
        ],
    )
    print(
        "WARNING: đây là fallback đoán để tránh rows=[]"
    )

    if debug:
        for event in aligned_events:
            print(
                f"    E{event['event_id']}: "
                f"frame={event['frame_id']} | "
                f"time={event['time']} | "
                f"score={event['score']:.4f}"
            )

    return [
        forced_candidate
    ]


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def run_query_item(
    query_item: Dict[str, Any],
    debug_retrieval: Optional[bool] = None,
    debug_video: Optional[bool] = None,
) -> Dict[str, Any]:
    if debug_retrieval is None:
        debug_retrieval = DEBUG_RETRIEVAL

    if debug_video is None:
        debug_video = DEBUG_VIDEO_PIPELINE

    analysis = prepare_analysis(query_item)
    retrieval_engine = get_retrieval_engine()

    # ========================================================
    # ========================================================
    # TASK-SPECIFIC RETRIEVAL
    # ========================================================

    task = str(
        analysis.get(
            "task",
            ""
        )
    ).upper()


    # ========================================================
    # TRAKE
    # ========================================================

    if task == "TRAKE":

        trake_retriever, trake_aligner = (
            get_trake_modules()
        )


        event_results = (
            trake_retriever.retrieve(
                analysis=analysis,
                debug=debug_retrieval
            )
        )


        retrieval_candidates = (
            trake_aligner.align(
                event_results=event_results,
                debug=debug_retrieval
            )
        )

        if not retrieval_candidates:
            retrieval_candidates = (
                run_relaxed_trake_alignment(
                    event_results=event_results,
                    debug=debug_retrieval,
                )
            )

        if not retrieval_candidates:
            retrieval_candidates = (
                build_forced_trake_candidates(
                    event_results=event_results,
                    analysis=analysis,
                    debug=debug_retrieval,
                )
            )


    # ========================================================
    # KIS / QA
    # ========================================================

    else:

        retrieval_candidates = (
            retrieval_engine.retrieve(

                analysis=analysis,

                top_k=(
                    RETRIEVAL_TOP_K
                ),

                top_n=(
                    RETRIEVAL_TOP_N
                ),

                debug=(
                    debug_retrieval
                )
            )
        )

    video_engine = get_video_pipeline_engine()
    video_result = video_engine.run(
        analysis=analysis,
        retrieval_candidates=retrieval_candidates,
        verify_top_n=VLM_VERIFY_TOP_N,
        debug=debug_video,
    )

    final_candidates = _reorder_after_vlm(
        retrieval_candidates=retrieval_candidates,
        video_candidates=video_result["candidates"],
        verify_top_n=VLM_VERIFY_TOP_N,
    )

    return {
        "query_item": query_item,
        "analysis": analysis,
        "retrieval_candidates": retrieval_candidates,
        "final_candidates": final_candidates,
        "task_result": video_result["task_result"],
    }


def run_query(
    raw_query: str,
    task: Optional[str] = None,
    debug_retrieval: Optional[bool] = None,
    debug_video: Optional[bool] = None,
):
    if not isinstance(raw_query, str):
        raise TypeError("raw_query phải là string.")

    raw_query = raw_query.strip()
    if not raw_query:
        raise ValueError("raw_query đang rỗng.")

    if task is None:
        temp_analysis = analyze(raw_query)
        task = str(temp_analysis["task"]).upper()

    task = task.upper()
    query_item = {
        "query_name": f"manual-{task.lower()}",
        "filename": f"manual-{task.lower()}.txt",
        "task": task,
        "text": raw_query,
    }

    return run_query_item(
        query_item,
        debug_retrieval=debug_retrieval,
        debug_video=debug_video,
    )


# ============================================================
# ANSWER SAFETY & ROW BUILDERS
# ============================================================

def normalize_qa_answer(answer: Any) -> str:
    if answer is None:
        return ""

    answer = str(answer)
    if len(answer) > 100:
        print("[WARNING] QA answer > 100 chars. Truncate về 100 chars.")
        answer = answer[:100]

    return answer


def build_kis_rows(result: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    candidates = result["final_candidates"]

    for candidate in candidates:
        if len(rows) >= SUBMISSION_KIS_MAX_LINES:
            break

        video_id = clean_video_id(candidate.get("video_id", ""))
        frame_id = resolve_candidate_frame(candidate)

        if not video_id or frame_id is None:
            continue

        rows.append([video_id, int(frame_id)])

    return rows


def build_qa_rows(result: Dict[str, Any]) -> List[List[Any]]:
    """
    QA is ANSWER-FIRST.

    The VLM is NOT asked to decide whether the whole scene satisfies every
    contextual detail of the query.

    Instead:
      - the original query tells it WHAT information/object/text/number/count
        it needs to inspect;
      - if it can see/read/count the requested target in the candidate scene,
        it returns the answer;
      - submission uses answer + answer_confidence, not vlm_verified.
    """
    rows = []

    for candidate in result["final_candidates"]:
        if len(rows) >= SUBMISSION_QA_MAX_LINES:
            break

        # Only VLM-processed candidates can contain a grounded QA answer.
        if candidate.get("vlm_processed") is not True:
            continue

        vlm_result = candidate.get("vlm_result")

        if not isinstance(vlm_result, dict):
            continue

        answer = normalize_qa_answer(
            vlm_result.get("answer")
        )

        if not answer:
            continue

        try:
            answer_confidence = float(
                vlm_result.get(
                    "answer_confidence",
                    vlm_result.get(
                        "confidence",
                        0.0,
                    ),
                )
            )
        except (TypeError, ValueError):
            answer_confidence = 0.0

        # Intentionally permissive. The goal is not to require the entire
        # contextual description to be verified; the goal is to use an answer
        # that the model can actually see/read/count in the scene.
        if answer_confidence < QA_MIN_ANSWER_CONFIDENCE:
            continue

        frame_id = resolve_candidate_frame(
            candidate
        )

        if frame_id is None:
            continue

        video_id = clean_video_id(
            candidate.get("video_id", "")
        )

        if not video_id:
            continue

        rows.append([
            video_id,
            int(frame_id),
            answer,
        ])

    # Defensive fallback: use VideoPipelineEngine's task_result if it already
    # produced a concrete answer.
    if not rows:
        task_result = result.get(
            "task_result",
            {},
        )

        answer = normalize_qa_answer(
            task_result.get("answer")
        )

        video_id = clean_video_id(
            task_result.get("video_id", "")
        )

        if answer and video_id:
            candidate = next(
                (
                    c
                    for c in result["final_candidates"]
                    if c.get("video_id") == video_id
                ),
                None,
            )

            if candidate is not None:
                frame_id = resolve_candidate_frame(
                    candidate
                )

                if frame_id is not None:
                    rows.append([
                        video_id,
                        int(frame_id),
                        answer,
                    ])


    # --------------------------------------------------------
    # FINAL DEFENSIVE QA FALLBACK
    # --------------------------------------------------------
    # The QA prompt is instructed to always return a best guess. This block is
    # only a last-resort guard so the whole submission package never aborts
    # because every VLM output was null/invalid.
    if not rows:
        analysis = result.get("analysis", {})
        answer_type = str(
            analysis.get("answer_type", "")
        ).strip().lower()

        # Choose the strongest available candidate for video/frame.
        candidates = result.get("final_candidates", [])

        fallback_candidate = (
            candidates[0]
            if candidates
            else None
        )

        if fallback_candidate is not None:
            video_id = clean_video_id(
                fallback_candidate.get("video_id", "")
            )

            frame_id = resolve_candidate_frame(
                fallback_candidate
            )

            # Generic non-empty emergency guess by answer type.
            # This should almost never be reached if the new QA prompt is active.
            if "yes" in answer_type or "boolean" in answer_type:
                fallback_answer = "yes"
            elif (
                "count" in answer_type
                or "number" in answer_type
                or "numeric" in answer_type
                or "quantity" in answer_type
            ):
                fallback_answer = "1"
            else:
                fallback_answer = "unknown"

            if video_id and frame_id is not None:
                rows.append([
                    video_id,
                    int(frame_id),
                    fallback_answer,
                ])

    return rows


def build_trake_rows(result: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    analysis = result["analysis"]

    expected_count = int(
        analysis.get(
            "expected_event_count",
            len(analysis.get("events", [])),
        )
    )

    if expected_count <= 0:
        raise RuntimeError(
            "TRAKE không xác định được số events."
        )

    resolver = get_frame_resolver()

    for candidate in result["final_candidates"]:
        if len(rows) >= SUBMISSION_TRAKE_MAX_LINES:
            break

        video_id = clean_video_id(
            candidate.get("video_id", "")
        )

        if not video_id:
            continue

        frames = None

        # ----------------------------------------------------
        # 1) Preferred: VLM event sequence
        # ----------------------------------------------------
        vlm_events = candidate.get(
            "vlm_events",
            [],
        )

        if (
            candidate.get("vlm_verified", False)
            and isinstance(vlm_events, list)
            and len(vlm_events) == expected_count
        ):
            event_times = []
            valid = True

            for event in vlm_events:
                if not event.get("matched", False):
                    valid = False
                    break

                original_time = event.get(
                    "original_time"
                )

                if original_time is None:
                    valid = False
                    break

                try:
                    event_times.append(
                        float(original_time)
                    )
                except (TypeError, ValueError):
                    valid = False
                    break

            if valid:
                valid = all(
                    event_times[i] < event_times[i + 1]
                    for i in range(
                        len(event_times) - 1
                    )
                )

            if valid:
                frames = resolver.nearest_sequence(
                    video_id=video_id,
                    event_times=event_times,
                )

        # ----------------------------------------------------
        # 2) Optional Retrieval alignment fallback
        # ----------------------------------------------------
        if (
            frames is None
            and SUBMISSION_TRAKE_ALLOW_RETRIEVAL_FALLBACK
        ):
            aligned_events = candidate.get(
                "aligned_events",
                [],
            )

            if (
                isinstance(aligned_events, list)
                and len(aligned_events) == expected_count
            ):
                try:
                    fallback_frames = [
                        int(event["frame_id"])
                        for event in aligned_events
                    ]

                    fallback_times = [
                        float(event["time"])
                        for event in aligned_events
                    ]

                    frame_order_ok = all(
                        fallback_frames[i] < fallback_frames[i + 1]
                        for i in range(
                            len(fallback_frames) - 1
                        )
                    )

                    time_order_ok = all(
                        fallback_times[i] < fallback_times[i + 1]
                        for i in range(
                            len(fallback_times) - 1
                        )
                    )

                    if frame_order_ok and time_order_ok:
                        frames = fallback_frames

                except (KeyError, TypeError, ValueError):
                    frames = None

        if frames is None or len(frames) != expected_count:
            continue

        rows.append([
            video_id,
            *[int(frame) for frame in frames],
        ])


    # --------------------------------------------------------
    # FINAL FORCED TRAKE ROW FALLBACK
    # --------------------------------------------------------
    # If VLM rejected everything and normal fallback is disabled/failed,
    # still emit the best-effort aligned sequence so the package does not abort.
    if (
        not rows
        and TRAKE_FORCED_BEST_GUESS_ENABLED
    ):
        source_candidates = (
            result.get(
                "retrieval_candidates",
                []
            )
            or result.get(
                "final_candidates",
                []
            )
        )

        for candidate in source_candidates:
            video_id = clean_video_id(
                candidate.get(
                    "video_id",
                    ""
                )
            )

            if not video_id:
                continue

            aligned_events = candidate.get(
                "aligned_events",
                []
            )

            if not isinstance(
                aligned_events,
                list
            ):
                continue

            if len(aligned_events) != expected_count:
                continue

            frames = []

            for event in aligned_events:
                try:
                    frames.append(
                        int(
                            event.get(
                                "frame_id"
                            )
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                ):
                    frames = []
                    break

            if len(frames) != expected_count:
                continue

            # Force strict monotonicity to satisfy BTC CSV format.
            fixed_frames = []

            for frame in frames:
                if not fixed_frames:
                    fixed_frames.append(
                        max(
                            int(frame),
                            0,
                        )
                    )
                else:
                    fixed_frames.append(
                        max(
                            int(frame),
                            fixed_frames[-1]
                            + TRAKE_FORCED_FRAME_STEP,
                        )
                    )

            rows.append([
                video_id,
                *fixed_frames,
            ])

            print(
                "[TRAKE FORCED ROW] "
                f"{video_id},"
                + ",".join(
                    str(frame)
                    for frame in fixed_frames
                )
            )

            break

    return rows


def build_submission_rows(result: Dict[str, Any]) -> List[List[Any]]:
    task = result["query_item"]["task"]
    if task == "KIS":
        return build_kis_rows(result)
    if task == "QA":
        return build_qa_rows(result)
    if task == "TRAKE":
        return build_trake_rows(result)

    raise ValueError(f"Unsupported task: {task}")


# ============================================================
# CSV & ZIP HANDLING
# ============================================================

def write_submission_csv(csv_path: str, rows: List[List[Any]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)


def validate_csv(
    csv_path: str,
    query_item: Dict[str, Any],
    expected_event_count: Optional[int] = None,
):
    task = query_item["task"]
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"{query_item['query_name']}: CSV đang rỗng.")

    if len(rows) > 100:
        raise ValueError(f"{query_item['query_name']}: quá 100 dòng.")

    video_pattern = re.compile(r"^L\d+_V\d+$", flags=re.IGNORECASE)

    for row_index, row in enumerate(rows, start=1):
        if task == "KIS":
            expected_columns = 2
        elif task == "QA":
            expected_columns = 3
        elif task == "TRAKE":
            if expected_event_count is None:
                raise ValueError("Thiếu expected_event_count.")
            expected_columns = 1 + expected_event_count
        else:
            raise ValueError(f"Unknown task: {task}")

        if len(row) != expected_columns:
            raise ValueError(
                f"{query_item['query_name']} row {row_index}: "
                f"expected {expected_columns} columns, got {len(row)}."
            )

        video_id = row[0]
        if video_id.lower().endswith(".mp4"):
            raise ValueError(f"{query_item['query_name']} row {row_index}: video không được có .mp4")

        if not video_pattern.match(video_id):
            raise ValueError(f"{query_item['query_name']} row {row_index}: video id không hợp lệ: {video_id}")

        frame_values = row[1:2] if task in ("KIS", "QA") else row[1:]
        parsed_frames = []

        for value in frame_values:
            try:
                frame = int(value)
            except ValueError:
                raise ValueError(f"{query_item['query_name']} row {row_index}: Frame ID không phải integer.")

            if frame < 0:
                raise ValueError(f"{query_item['query_name']} row {row_index}: Frame ID < 0.")

            parsed_frames.append(frame)

        if task == "TRAKE" and any(parsed_frames[i] >= parsed_frames[i + 1] for i in range(len(parsed_frames) - 1)):
            raise ValueError(f"{query_item['query_name']} row {row_index}: TRAKE frame IDs không tăng theo thời gian.")

        if task == "QA":
            answer = row[2]
            if answer == "":
                raise ValueError(f"{query_item['query_name']} row {row_index}: Answer rỗng.")
            if len(answer) > 100:
                raise ValueError(f"{query_item['query_name']} row {row_index}: Answer > 100 ký tự.")


def create_submission_zip(submission_dir: str, output_zip_path: str):
    parent = os.path.dirname(output_zip_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)

    csv_files = sorted(
        Path(submission_dir).glob("*.csv"),
        key=lambda p: _natural_sort_key(p.name),
    )

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in csv_files:
            arcname = f"submission/{csv_path.name}"
            zf.write(str(csv_path), arcname=arcname)

    return output_zip_path


def validate_submission_zip(zip_path: str, query_items: List[Dict[str, Any]]):
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Output không phải ZIP hợp lệ.")

    expected_names = {f"submission/{item['query_name']}.csv" for item in query_items}

    with zipfile.ZipFile(zip_path, "r") as zf:
        actual_names = {name for name in zf.namelist() if name.lower().endswith(".csv")}

    missing = expected_names - actual_names
    if missing:
        raise ValueError("Submission ZIP thiếu CSV:\n" + "\n".join(sorted(missing)))

    print("\n==========================================")
    print("SUBMISSION ZIP VALID")
    print("==========================================")
    print(f"CSV files : {len(actual_names)}")
    print(f"ZIP       : {zip_path}")
    print("==========================================")


# ============================================================
# RUN WHOLE QUERY PACKAGE
# ============================================================

def run_query_package(
    query_source: Optional[str] = None,
    output_zip_path: Optional[str] = None,
    start_from: int = 1,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run all queries or resume from a 1-based query index.

    Example:
        run_query_package(start_from=16, resume=True)

    With resume=True, existing CSVs for queries before start_from are preserved.
    """
    if query_source is None:
        query_source = QUERY_PACKAGE_PATH

    if output_zip_path is None:
        output_zip_path = SUBMISSION_ZIP_PATH

    query_items = load_query_package(query_source)

    if start_from < 1 or start_from > len(query_items):
        raise ValueError(
            f"start_from phải nằm trong [1, {len(query_items)}], "
            f"nhận được {start_from}."
        )

    work_root = os.path.abspath(SUBMISSION_WORK_ROOT)
    submission_dir = os.path.join(work_root, "submission")

    if not resume:
        if os.path.isdir(submission_dir):
            shutil.rmtree(submission_dir)

        os.makedirs(submission_dir, exist_ok=True)

    else:
        os.makedirs(submission_dir, exist_ok=True)

        missing_previous = []

        for item in query_items[:start_from - 1]:
            csv_path = os.path.join(
                submission_dir,
                f"{item['query_name']}.csv",
            )

            if not os.path.isfile(csv_path):
                missing_previous.append(item["query_name"])

        if missing_previous:
            message = (
                "Không thể resume vì thiếu CSV của các query trước:\n"
                + "\n".join(
                    f"    {name}"
                    for name in missing_previous
                )
            )
            raise RuntimeError(message)

        print("\n==========================================")
        print("RESUME MODE")
        print("==========================================")
        print(f"Start from query : {start_from}")
        print(f"Preserved CSVs   : {start_from - 1}")
        print("==========================================")

    completed = []
    errors = []

    if resume and start_from > 1:
        for item in query_items[:start_from - 1]:
            csv_path = os.path.join(
                submission_dir,
                f"{item['query_name']}.csv",
            )

            try:
                with open(
                    csv_path,
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    row_count = sum(1 for _ in csv.reader(f))
            except Exception:
                row_count = 0

            completed.append({
                "query_name": item["query_name"],
                "task": item["task"],
                "rows": row_count,
                "csv_path": csv_path,
                "preserved": True,
            })

    for index in range(start_from - 1, len(query_items)):
        query_item = query_items[index]
        display_index = index + 1

        query_name = query_item["query_name"]
        task = query_item["task"]

        print("\n\n##########################################")
        print(f"QUERY {display_index}/{len(query_items)}")
        print(f"NAME : {query_name}")
        print(f"TASK : {task}")
        print("##########################################")

        try:
            result = run_query_item(query_item)
            rows = build_submission_rows(result)

            if not rows:
                raise RuntimeError(
                    f"{query_name}: không tạo được prediction hợp lệ."
                )

            rows = rows[:100]

            csv_path = os.path.join(
                submission_dir,
                f"{query_name}.csv",
            )

            write_submission_csv(csv_path, rows)

            expected_event_count = None

            if task == "TRAKE":
                expected_event_count = int(
                    result["analysis"].get(
                        "expected_event_count",
                        len(result["analysis"].get("events", [])),
                    )
                )

            validate_csv(
                csv_path=csv_path,
                query_item=query_item,
                expected_event_count=expected_event_count,
            )

            print(
                f"[OK] {query_name}.csv\n"
                f"     rows = {len(rows)}"
            )

            completed.append({
                "query_name": query_name,
                "task": task,
                "rows": len(rows),
                "csv_path": csv_path,
            })

        except Exception as error:
            print(f"\n[ERROR] {query_name}\n{error}")

            errors.append({
                "query_name": query_name,
                "task": task,
                "error": str(error),
            })

            if SUBMISSION_STRICT:
                break

        finally:
            if CLEAN_CACHE_AFTER_EACH_QUERY:
                cleanup_query_cache()

    if errors and SUBMISSION_STRICT:
        print("\n==========================================")
        print("SUBMISSION ABORTED")
        print("==========================================")
        print("Có query bị lỗi. Không tạo submission.zip để tránh nộp nhầm.")
        print("Query lỗi:")

        for item in errors:
            print(f"    {item['query_name']}: {item['error']}")

        raise RuntimeError("Submission validation failed.")

    csv_count = len(list(Path(submission_dir).glob("*.csv")))

    if SUBMISSION_STRICT and csv_count != len(query_items):
        raise RuntimeError(
            f"Expected {len(query_items)} CSV, got {csv_count}."
        )

    create_submission_zip(
        submission_dir=submission_dir,
        output_zip_path=output_zip_path,
    )

    validate_submission_zip(
        zip_path=output_zip_path,
        query_items=query_items,
    )

    return {
        "query_count": len(query_items),
        "completed": completed,
        "errors": errors,
        "submission_dir": submission_dir,
        "submission_zip": output_zip_path,
        "start_from": start_from,
        "resume": resume,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    final_result = run_query_package()
    print("\n==========================================")
    print("READY FOR SUBMISSION")
    print("==========================================")
    print("File:")
    print(final_result["submission_zip"])
    print("==========================================")