
from typing import (
    Dict,
    Any,
    List,
    Optional
)

from video_pipeline.video_locator import (
    VideoLocator
)

from video_pipeline.video_cache import (
    VideoCache
)

from video_pipeline.segment_builder import (
    SegmentBuilder
)

from video_pipeline.vlm_runner import (
    VLMRunner
)

from video_pipeline.vlm_reranker import (
    VLMReranker
)


class VideoPipelineEngine:
    """
    Pipeline:

        Retrieval Candidates
            ↓
        VideoLocator
            ↓
        VideoCache
            ↓
        SegmentBuilder
            ↓
        VLMRunner
            ↓
        VLMReranker
            ↓
        Final candidates
    """

    def __init__(
        self,

        # ====================================================
        # VIDEO DATA
        # ====================================================

        videos_root: str,

        cache_root: str = (
            "/content/video_cache"
        ),

        segment_root: str = (
            "/content/video_cache/segments"
        ),


        # ====================================================
        # SEGMENT
        # ====================================================

        padding_before: float = 5.0,

        padding_after: float = 5.0,

        min_segment_duration: float = 6.0,

        max_segment_duration: float = 30.0,


        # ====================================================
        # VLM
        # ====================================================

        enable_vlm: bool = True,

        vlm_model_name: str = (
            "Qwen/Qwen3-VL-4B-Instruct"
        ),

        vlm_device_map: str = "auto",

        vlm_dtype: str = "auto",

        vlm_fps_kis: float = 2.0,

        vlm_fps_qa: float = 2.0,

        vlm_fps_trake: float = 4.0,

        vlm_max_new_tokens: int = 512,


        # ====================================================
        # VLM RERANK
        # ====================================================

        vlm_positive_boost: float = 0.50,

        vlm_reject_penalty: float = 0.80,

        vlm_minimum_factor: float = 0.10,

        vlm_match_threshold: float = 0.50,
        # ====================================================
        # HOW MANY VIDEOS VLM CHECKS
        # ====================================================

        verify_top_n: int = 5
    ):

        if verify_top_n <= 0:

            raise ValueError(
                "verify_top_n phải > 0."
            )

        self.verify_top_n = int(
            verify_top_n
        )

        self.enable_vlm = bool(
            enable_vlm
        )

        # ====================================================
        # LOCATOR
        # ====================================================

        self.locator = VideoLocator(
            videos_root=videos_root
        )

        # ====================================================
        # CACHE
        # ====================================================

        self.cache = VideoCache(
            cache_root=cache_root
        )

        # ====================================================
        # SEGMENT
        # ====================================================

        self.segment_builder = (
            SegmentBuilder(

                segment_root=(
                    segment_root
                ),

                padding_before=(
                    padding_before
                ),

                padding_after=(
                    padding_after
                ),

                min_segment_duration=(
                    min_segment_duration
                ),

                max_segment_duration=(
                    max_segment_duration
                )
            )
        )

        # ====================================================
        # RERANKER
        # ====================================================

        self.reranker = (
            VLMReranker(

                positive_boost=(
                    vlm_positive_boost
                ),

                reject_penalty=(
                    vlm_reject_penalty
                ),

                minimum_factor=(
                    vlm_minimum_factor
                ),

                match_threshold=(
                    vlm_match_threshold
                )
            )
        )

        # ====================================================
        # STORE VLM CONFIG
        # ====================================================

        self.vlm_config = {

            "model_name":
                vlm_model_name,

            "device_map":
                vlm_device_map,

            "dtype":
                vlm_dtype,

            "fps_kis":
                vlm_fps_kis,

            "fps_qa":
                vlm_fps_qa,

            "fps_trake":
                vlm_fps_trake,

            "max_new_tokens":
                vlm_max_new_tokens
        }

        # Lazy load
        self.vlm_runner = None


    # ========================================================
    # LOAD VLM ONLY WHEN NEEDED
    # ========================================================

    def _get_vlm_runner(
        self
    ):

        if not self.enable_vlm:

            return None

        if self.vlm_runner is None:

            self.vlm_runner = (
                VLMRunner(
                    **self.vlm_config
                )
            )

        return self.vlm_runner


    # ========================================================
    # BUILD TASK RESULT
    # ========================================================

    def build_task_result(
        self,
        analysis: Dict[str, Any],
        candidates: List[
            Dict[str, Any]
        ]
    ) -> Dict[str, Any]:

        task = str(
            analysis.get(
                "task",
                ""
            )
        ).upper()

        if not candidates:

            return {
                "task":
                    task,

                "success":
                    False,

                "reason":
                    "No candidates."
            }

        # ====================================================
        # KIS
        # ====================================================

        if task == "KIS":

            best = candidates[0]

            best_time = best.get(
                "vlm_best_time"
            )

            if best_time is None:

                best_time = best.get(
                    "candidate_time"
                )

            return {
                "task":
                    "KIS",

                "success":
                    True,

                "video_id":
                    best.get(
                        "video_id"
                    ),

                "time":
                    best_time,

                "frame":
                    best.get(
                        "candidate_frame"
                    ),

                "verified":
                    best.get(
                        "vlm_verified",
                        False
                    ),

                "confidence":
                    best.get(
                        "vlm_confidence",
                        0.0
                    ),

                "score":
                    best.get(
                        "final_score",
                        best.get(
                            "score"
                        )
                    )
            }

        # ====================================================
        # QA
        # ====================================================

        if task == "QA":

            # candidate cao nhất có answer
            for candidate in candidates:

                vlm_result = candidate.get(
                    "vlm_result"
                )

                if not isinstance(
                    vlm_result,
                    dict
                ):
                    continue

                answer = vlm_result.get(
                    "answer"
                )

                if answer is None:
                    continue

                return {
                    "task":
                        "QA",

                    "success":
                        True,

                    "answer":
                        answer,

                    "video_id":
                        candidate.get(
                            "video_id"
                        ),

                    "time":
                        candidate.get(
                            "vlm_best_time",
                            candidate.get(
                                "candidate_time"
                            )
                        ),

                    "confidence":
                        candidate.get(
                            "vlm_confidence",
                            0.0
                        )
                }

            return {
                "task":
                    "QA",

                "success":
                    False,

                "answer":
                    None,

                "reason":
                    "No verified answer."
            }

        # ====================================================
        # TRAKE
        # ====================================================

        if task == "TRAKE":

            best = candidates[0]

            return {
                "task":
                    "TRAKE",

                "success":
                    True,

                "video_id":
                    best.get(
                        "video_id"
                    ),

                "verified":
                    best.get(
                        "vlm_verified",
                        False
                    ),

                "confidence":
                    best.get(
                        "vlm_confidence",
                        0.0
                    ),

                "events":
                    best.get(
                        "vlm_events",
                        []
                    )
            }

        raise ValueError(
            f"Unsupported task: {task}"
        )


    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        analysis: Dict[str, Any],
        retrieval_candidates: List[
            Dict[str, Any]
        ],
        verify_top_n: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Any]:

        if not retrieval_candidates:

            return {
                "candidates": [],
                "task_result": (
                    self.build_task_result(
                        analysis,
                        []
                    )
                )
            }

        if verify_top_n is None:

            verify_top_n = (
                self.verify_top_n
            )

        verify_top_n = min(
            int(
                verify_top_n
            ),
            len(
                retrieval_candidates
            )
        )

        # ====================================================
        # SAVE ORIGINAL RETRIEVAL SCORES
        # ====================================================

        base_candidates = []

        for candidate in retrieval_candidates:

            item = dict(
                candidate
            )

            item[
                "retrieval_score"
            ] = float(
                candidate.get(
                    "score",
                    0.0
                )
            )

            item[
                "vlm_processed"
            ] = False

            base_candidates.append(
                item
            )

        selected = (
            base_candidates[
                :verify_top_n
            ]
        )

        if debug:

            print(
                "\n"
                "=========================================="
            )

            print(
                "VIDEO PIPELINE"
            )

            print(
                "=========================================="
            )

            print(
                f"Retrieval candidates: "
                f"{len(base_candidates)}"
            )

            print(
                f"VLM verify top: "
                f"{verify_top_n}"
            )

        # ====================================================
        # STEP 1 - LOCATE
        # ====================================================

        located = (
            self.locator
            .locate_candidates(
                selected
            )
        )

        if debug:

            found = sum(
                1
                for c in located
                if c.get(
                    "video_found"
                )
            )

            print(
                f"\n[VIDEO 1] Located: "
                f"{found}/{len(located)}"
            )

        # ====================================================
        # STEP 2 - CACHE VIDEO
        # ====================================================

        cached = (
            self.cache
            .cache_candidates(
                located
            )
        )

        if debug:

            ready = sum(
                1
                for c in cached
                if c.get(
                    "video_cached"
                )
            )

            print(
                f"[VIDEO 2] Cached: "
                f"{ready}/{len(cached)}"
            )

        # ====================================================
        # STEP 3 - SEGMENTS
        # ====================================================

        segmented = (
            self.segment_builder
            .build_segments(
                cached
            )
        )

        if debug:

            ready = sum(
                1
                for c in segmented
                if c.get(
                    "segment_ready"
                )
            )

            print(
                f"[VIDEO 3] Segments: "
                f"{ready}/{len(segmented)}"
            )

        # ====================================================
        # STEP 4 - VLM
        # ====================================================

        if self.enable_vlm:

            vlm_runner = (
                self._get_vlm_runner()
            )

            ready_segments = [

                candidate

                for candidate
                in segmented

                if candidate.get(
                    "segment_ready"
                )
            ]

            vlm_results = (
                vlm_runner
                .run_candidates(
                    analysis=analysis,
                    candidates=ready_segments,
                    debug=debug
                )
            )

            vlm_by_video = {

                item[
                    "video_id"
                ]:
                    item

                for item
                in vlm_results
            }

        else:

            vlm_by_video = {}

        # ====================================================
        # PRESERVE SEGMENT ERRORS
        # ====================================================

        processed_by_video = {

            item[
                "video_id"
            ]:
                item

            for item
            in segmented
        }

        # VLM results override segmented candidate
        processed_by_video.update(
            vlm_by_video
        )

        # ====================================================
        # MERGE TOP VERIFIED + REMAINING RETRIEVAL
        # ====================================================

        merged_candidates = []

        for candidate in base_candidates:

            video_id = candidate.get(
                "video_id"
            )

            if video_id in processed_by_video:

                merged_candidates.append(
                    processed_by_video[
                        video_id
                    ]
                )

            else:

                merged_candidates.append(
                    candidate
                )

        # ====================================================
        # STEP 5 - VLM RERANK
        # ====================================================

        final_candidates = (
            self.reranker.rank(
                merged_candidates
            )
        )

        # ====================================================
        # TASK OUTPUT
        # ====================================================

        task_result = (
            self.build_task_result(
                analysis,
                final_candidates
            )
        )

        if debug:

            print(
                "\n[VIDEO 4] FINAL RANKING"
            )

            for candidate in (
                final_candidates[:10]
            ):

                print(
                    f"#{candidate['rank']} "
                    f"{candidate['video_id']} | "
                    f"retrieval="
                    f"{candidate['retrieval_score']:.4f} | "
                    f"VLM="
                    f"{candidate.get('vlm_confidence', 0):.3f} | "
                    f"{candidate.get('vlm_rerank_status')} | "
                    f"final="
                    f"{candidate['final_score']:.4f}"
                )

        return {
            "candidates":
                final_candidates,

            "task_result":
                task_result
        }
