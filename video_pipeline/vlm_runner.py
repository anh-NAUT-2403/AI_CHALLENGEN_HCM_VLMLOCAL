# video_pipeline/vlm_runner.py

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from video_pipeline.vlm_prompts import SYSTEM_PROMPT, build_vlm_prompt


class VLMRunner:
    """
    Chạy Qwen3-VL trên candidate video segment.

    Input:
        analysis
        candidate có segment_path

    Output:
        candidate cũ
        +
        vlm_result
        vlm_verified
        vlm_confidence
        vlm_best_time
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device_map: str = "auto",
        dtype: str = "auto",
        fps_kis: float = 2.0,
        fps_qa: float = 2.0,
        fps_trake: float = 4.0,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype

        self.fps_by_task = {
            "KIS": float(fps_kis),
            "QA": float(fps_qa),
            "TRAKE": float(fps_trake),
        }

        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens phải > 0.")

        print("\n==========================================")
        print("LOADING VLM")
        print("==========================================")
        print(f"Model: {self.model_name}")

        # Model & Processor initialization
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device_map,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("[INFO] VLM READY")

    # ========================================================
    # FPS
    # ========================================================

    def get_task_fps(self, task: str) -> float:
        task = str(task).upper()
        return self.fps_by_task.get(task, 2.0)

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return min(max(value, 0.0), 1.0)

    # ========================================================
    # JSON PARSER
    # ========================================================

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("VLM output phải là string.")

        text = text.strip()

        # Direct JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove markdown fences
        cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            candidate_json = cleaned[start : end + 1]
            try:
                return json.loads(candidate_json)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Không parse được JSON từ VLM:\n{text}")

    # ========================================================
    # MESSAGE BUILDER
    # ========================================================

    def _build_messages(
        self,
        analysis: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        segment_path = candidate.get("segment_path")
        if not segment_path:
            raise ValueError("candidate thiếu segment_path.")

        if not os.path.isfile(segment_path):
            raise FileNotFoundError(f"Không tìm thấy segment: {segment_path}")

        task = analysis.get("task", "KIS")
        fps = self.get_task_fps(task)
        prompt = build_vlm_prompt(analysis=analysis, candidate=candidate)

        # Qwen-VL accepts file:// URI for local video paths
        video_uri = Path(segment_path).resolve().as_uri()

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_uri,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

    # ========================================================
    # TRAKE NORMALIZER
    # ========================================================

    def _normalize_trake_result(
        self,
        analysis: Dict[str, Any],
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_events = analysis.get("events", [])
        returned_events = parsed.get("events", [])

        if not isinstance(returned_events, list):
            returned_events = []

        # Map returned events by event_id
        returned_map = {}
        for item in returned_events:
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if event_id is not None:
                returned_map[str(event_id)] = item

        # Normalize exactly N required events
        normalized = []
        for index, expected in enumerate(expected_events, start=1):
            event_id = expected.get("event_id", index)
            item = returned_map.get(str(event_id), {})

            matched = bool(item.get("matched", False))
            relative_time = item.get("relative_time")

            if relative_time is not None:
                try:
                    relative_time = float(relative_time)
                except (TypeError, ValueError):
                    relative_time = None

            confidence = item.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            # matched=True nhưng không có timestamp thì tính là chưa match
            if relative_time is None:
                matched = False

            normalized.append({
                "event_id": event_id,
                "matched": matched,
                "relative_time": relative_time,
                "confidence": confidence,
            })

        parsed["events"] = normalized

        # Check complete sequence & temporal ordering
        complete = (
            len(normalized) == len(expected_events)
            and len(expected_events) > 0
            and all(event["matched"] for event in normalized)
        )

        ordered = False
        if complete:
            times = [event["relative_time"] for event in normalized]
            ordered = all(times[i] < times[i + 1] for i in range(len(times) - 1))

        parsed["verified"] = bool(complete and ordered)
        return parsed

    # ========================================================
    # RUN ONE CANDIDATE
    # ========================================================

    def run_candidate(
        self,
        analysis: Dict[str, Any],
        candidate: Dict[str, Any],
        debug: bool = False,
    ) -> Dict[str, Any]:
        messages = self._build_messages(analysis, candidate)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        (
            image_inputs,
            video_inputs,
            video_kwargs,
        ) = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            videos = None
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if debug:
            print("\n========== RAW VLM ==========")
            print(output_text)

        parsed = self._extract_json(output_text)

        task = str(analysis.get("task", "")).upper()
        if task == "TRAKE":
            parsed = self._normalize_trake_result(analysis, parsed)

        confidence = self._normalize_confidence(parsed.get("confidence", 0.0))
        verified = bool(parsed.get("verified", False))

        result = dict(candidate)
        result.update({
            "vlm_processed": True,
            "vlm_model": self.model_name,
            "vlm_task": analysis.get("task"),
            "vlm_result": parsed,
            "vlm_verified": verified,
            "vlm_confidence": confidence,
        })

        # Calculate best timestamp for KIS / QA
        relative_time = parsed.get("best_relative_time")
        if relative_time is not None:
            try:
                segment_start = float(candidate.get("segment_start", 0.0))
                result["vlm_best_time"] = segment_start + float(relative_time)
            except (TypeError, ValueError):
                result["vlm_best_time"] = None
        else:
            result["vlm_best_time"] = None

        # Calculate event timestamps for TRAKE
        if task == "TRAKE":
            converted_events = []
            for event in parsed.get("events", []):
                new_event = dict(event)
                relative = event.get("relative_time")
                if relative is not None:
                    try:
                        segment_start = float(candidate.get("segment_start", 0.0))
                        new_event["original_time"] = segment_start + float(relative)
                    except (TypeError, ValueError):
                        new_event["original_time"] = None
                else:
                    new_event["original_time"] = None
                converted_events.append(new_event)

            result["vlm_events"] = converted_events

        return result

    # ========================================================
    # RUN BATCH OF CANDIDATES
    # ========================================================

    def run_candidates(
        self,
        analysis: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, candidate in enumerate(candidates, start=1):
            video_id = candidate.get("video_id", "UNKNOWN")
            print(f"\n[VLM] {i}/{len(candidates)} {video_id}")

            try:
                result = self.run_candidate(
                    analysis=analysis,
                    candidate=candidate,
                    debug=debug,
                )
            except Exception as error:
                result = dict(candidate)
                result.update({
                    "vlm_processed": True,
                    "vlm_verified": False,
                    "vlm_confidence": 0.0,
                    "vlm_result": None,
                    "vlm_error": str(error),
                })

            results.append(result)

        return results