import os
import subprocess
from typing import Dict, Any, Optional, List


class SegmentBuilder:
    """
    Tạo clip ngắn từ video candidate.

    Input:
        local_video_path
        candidate_time hoặc temporal_window

    Output:
        local segment .mp4

    Baseline:
        padding_before = 5s
        padding_after  = 5s

    Ví dụ:
        temporal window: 43.6 -> 46.08
        output segment: 38.6 -> 51.08
    """

    def __init__(
        self,
        segment_root: str = "/content/video_cache/segments",
        padding_before: float = 5.0,
        padding_after: float = 5.0,
        min_segment_duration: float = 6.0,
        max_segment_duration: float = 30.0
    ):
        self.segment_root = os.path.abspath(segment_root)
        os.makedirs(self.segment_root, exist_ok=True)

        if padding_before < 0:
            raise ValueError("padding_before phải >= 0.")
        if padding_after < 0:
            raise ValueError("padding_after phải >= 0.")
        if min_segment_duration <= 0:
            raise ValueError("min_segment_duration phải > 0.")
        if max_segment_duration <= 0:
            raise ValueError("max_segment_duration phải > 0.")
        if max_segment_duration < min_segment_duration:
            raise ValueError("max_segment_duration phải >= min_segment_duration.")

        self.padding_before = float(padding_before)
        self.padding_after = float(padding_after)
        self.min_segment_duration = float(min_segment_duration)
        self.max_segment_duration = float(max_segment_duration)

    # ========================================================
    # GET VIDEO DURATION
    # ========================================================
    def get_video_duration(self, video_path: str) -> float:
        """Dùng ffprobe để lấy duration video."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe thất bại:\n{result.stderr}")
        try:
            return float(result.stdout.strip())
        except ValueError:
            raise RuntimeError("Không đọc được duration video.")

    # ========================================================
    # BUILD TIME WINDOW
    # ========================================================
    def build_window(
        self, candidate: Dict[str, Any], video_duration: float
    ) -> Dict[str, float]:
        """
        Xây segment time window từ candidate.
        Ưu tiên: temporal_window. Nếu không có: candidate_time.
        """
        temporal_window = candidate.get("temporal_window")

        # Case 1: Temporal Window
        if isinstance(temporal_window, dict):
            start_time = float(temporal_window.get("start_time", candidate.get("candidate_time", 0.0)))
            end_time = float(temporal_window.get("end_time", candidate.get("candidate_time", start_time)))
        # Case 2: Only Candidate Time
        else:
            candidate_time = float(candidate.get("candidate_time", 0.0))
            start_time = candidate_time
            end_time = candidate_time

        # Add Padding
        segment_start = max(0.0, start_time - self.padding_before)
        segment_end = min(video_duration, end_time + self.padding_after)

        # Minimum duration
        duration = segment_end - segment_start
        if duration < self.min_segment_duration:
            center = (segment_start + segment_end) / 2.0
            half = self.min_segment_duration / 2.0
            segment_start = max(0.0, center - half)
            segment_end = min(video_duration, center + half)

            # Nếu đụng biên video
            if segment_end - segment_start < self.min_segment_duration:
                if segment_start <= 0:
                    segment_end = min(video_duration, self.min_segment_duration)
                elif segment_end >= video_duration:
                    segment_start = max(0.0, video_duration - self.min_segment_duration)

        # Maximum duration
        duration = segment_end - segment_start
        if duration > self.max_segment_duration:
            candidate_time = float(candidate.get("candidate_time", (segment_start + segment_end) / 2.0))
            half = self.max_segment_duration / 2.0
            segment_start = max(0.0, candidate_time - half)
            segment_end = min(video_duration, candidate_time + half)

            # Giữ max duration nếu chạm biên
            if segment_end - segment_start < self.max_segment_duration:
                if segment_start <= 0:
                    segment_end = min(video_duration, self.max_segment_duration)
                elif segment_end >= video_duration:
                    segment_start = max(0.0, video_duration - self.max_segment_duration)

        return {
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "segment_duration": float(segment_end - segment_start)
        }

    # ========================================================
    # OUTPUT PATH
    # ========================================================
    def get_segment_path(self, video_id: str, start_time: float, end_time: float) -> str:
        filename = f"{video_id}__{start_time:.2f}__{end_time:.2f}.mp4"
        return os.path.join(self.segment_root, filename)

    # ========================================================
    # CUT VIDEO
    # ========================================================
    def cut_segment(
        self, video_path: str, output_path: str, start_time: float, end_time: float, force: bool = False
    ) -> str:
        if os.path.isfile(output_path) and not force:
            return output_path

        duration = end_time - start_time
        if duration <= 0:
            raise ValueError("Segment duration phải > 0.")

        command = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
            output_path
        ]

        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg thất bại:\n{result.stderr}")

        if not os.path.isfile(output_path):
            raise RuntimeError("Không tạo được segment.")
        if os.path.getsize(output_path) <= 0:
            raise RuntimeError("Segment có size = 0.")

        verify_command = ["ffmpeg", "-v", "error", "-i", output_path, "-f", "null", "-"]
        verify_result = subprocess.run(
            verify_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if verify_result.returncode != 0:
            raise RuntimeError(f"Segment đã được tạo nhưng không decode được:\n{verify_result.stderr}")

        return output_path

    # ========================================================
    # BUILD ONE CANDIDATE
    # ========================================================
    def build_candidate_segment(
        self, candidate: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        video_id = candidate.get("video_id")
        video_path = candidate.get("local_video_path")

        if not video_id:
            raise ValueError("candidate thiếu video_id.")
        if not video_path:
            raise ValueError("candidate thiếu local_video_path.")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Không tìm thấy local video: {video_path}")

        video_duration = self.get_video_duration(video_path)
        window = self.build_window(candidate=candidate, video_duration=video_duration)

        start_time = window["segment_start"]
        end_time = window["segment_end"]

        segment_path = self.get_segment_path(video_id=video_id, start_time=start_time, end_time=end_time)
        segment_path = self.cut_segment(
            video_path=video_path, output_path=segment_path,
            start_time=start_time, end_time=end_time, force=force
        )

        result = dict(candidate)
        result.update({
            "video_duration": float(video_duration),
            "segment_start": float(start_time),
            "segment_end": float(end_time),
            "segment_duration": float(end_time - start_time),
            "segment_path": segment_path,
            "segment_ready": True
        })
        return result

    # ========================================================
    # BUILD MANY
    # ========================================================
    def build_segments(
        self, candidates: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not isinstance(candidates, list):
            raise TypeError("candidates phải là list.")

        selected = candidates[:limit] if limit is not None and limit > 0 else candidates
        results = []

        for i, candidate in enumerate(selected, start=1):
            video_id = candidate.get("video_id", "UNKNOWN")
            print(f"\n[SegmentBuilder] {i}/{len(selected)} {video_id}")

            try:
                result = self.build_candidate_segment(candidate)
            except Exception as error:
                result = dict(candidate)
                result.update({
                    "segment_ready": False,
                    "segment_path": None,
                    "segment_error": str(error)
                })
            results.append(result)

        return results