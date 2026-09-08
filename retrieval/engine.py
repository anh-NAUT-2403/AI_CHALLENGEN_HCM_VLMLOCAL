# retrieval/engine.py

from typing import Any, Dict, List, Optional

from retrieval.data_store import RetrievalDataStore
from retrieval.encoder import CLIPTextEncoder
from retrieval.fusion import FusionScorer
from retrieval.hit_mapper import HitMapper
from retrieval.hit_pool import HitPoolBuilder
from retrieval.object_enricher import ObjectEnricher
from retrieval.object_matcher import ObjectMatcher
from retrieval.object_store import ObjectStore
from retrieval.ranker import CandidateRanker
from retrieval.searcher import FaissSearcher
from retrieval.temporal import TemporalClusterer
from retrieval.video_grouper import VideoGrouper


class RetrievalEngine:
    """High-level Retrieval Pipeline.

    Flow:
        Analyzer output
            ↓
        expanded_queries + entities
            ↓
        CLIP text encode
            ↓
        FAISS Top-K / query
            ↓
        metadata mapping
            ↓
        object enrichment
            ↓
        unified hit pool
            ↓
        exact keyframe grouping
            ↓
        group by video
            ↓
        temporal clustering
            ↓
        semantic fusion
            ↓
        object boost
            ↓
        candidate ranking
            ↓
        Top-N videos

    Object metadata là optional.
    Nếu object_zip_path=None:
        pipeline tự động chạy CLIP-only.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        # Data
        index_path: str,
        metadata_path: str,
        object_zip_path: Optional[str] = None,
        # Retrieval
        top_k: int = 100,
        top_n: int = 10,
        # Temporal
        max_gap_seconds: float = 10.0,
        max_cluster_duration: float = 30.0,
        # Query fusion
        best_weight: float = 0.60,
        second_weight: float = 0.30,
        coverage_weight: float = 0.10,
        # Object
        object_detection_threshold: float = 0.30,
        object_semantic_threshold: float = 0.83,
        object_boost_weight: float = 0.15,
        # CLIP
        clip_model_name: str = "ViT-B/32",
        device: Optional[str] = None,
    ):
        print("\n" + "=" * 42)
        print("INITIALIZING RETRIEVAL ENGINE")
        print("=" * 42)

        # ====================================================
        # VALIDATE CONFIG
        # ====================================================
        if top_k <= 0:
            raise ValueError("top_k phải > 0.")
        if top_n <= 0:
            raise ValueError("top_n phải > 0.")
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds phải > 0.")
        if max_cluster_duration <= 0:
            raise ValueError("max_cluster_duration phải > 0.")

        self.top_k = int(top_k)
        self.top_n = int(top_n)

        # ====================================================
        # 1. DATA STORE
        # ====================================================
        print("\n[1] Loading retrieval data...")
        self.store = RetrievalDataStore(
            index_path=index_path, metadata_path=metadata_path
        )
        self.store.load()

        # ====================================================
        # 2. CLIP TEXT ENCODER
        # ====================================================
        print("\n[2] Loading CLIP encoder...")
        self.encoder = CLIPTextEncoder(
            model_name=clip_model_name, device=device
        )

        # ====================================================
        # DIMENSION CHECK
        # ====================================================
        # Encode một text rất ngắn để đảm bảo dimension tương thích với FAISS
        test_embedding = self.encoder.encode("test")
        if test_embedding.shape[0] != self.store.index.d:
            raise ValueError(
                "CLIP / FAISS dimension mismatch: "
                f"CLIP={test_embedding.shape[0]}, FAISS={self.store.index.d}"
            )

        # ====================================================
        # 3. FAISS SEARCHER
        # ====================================================
        print("\n[3] Initializing FAISS searcher...")
        self.searcher = FaissSearcher(self.store.index)

        # ====================================================
        # 4. HIT MAPPER
        # ====================================================
        self.mapper = HitMapper(self.store.metadata)

        # ====================================================
        # 5. OBJECT PIPELINE
        # ====================================================
        self.object_enabled = object_zip_path is not None

        if self.object_enabled:
            print("\n[4] Initializing object metadata...")
            self.object_store = ObjectStore(object_zip_path)
            self.object_matcher = ObjectMatcher(
                encoder=self.encoder,
                min_detection_score=object_detection_threshold,
                semantic_threshold=object_semantic_threshold,
            )
            self.object_enricher = ObjectEnricher(
                object_store=self.object_store,
                object_matcher=self.object_matcher,
                detection_threshold=object_detection_threshold,
            )
            print("[INFO] Object boost ENABLED")
        else:
            self.object_store = None
            self.object_matcher = None
            self.object_enricher = None
            print("\n[4] Object boost DISABLED")

        # ====================================================
        # 6. HIT POOL
        # ====================================================
        self.hit_pool_builder = HitPoolBuilder()

        # ====================================================
        # 7. VIDEO GROUPER
        # ====================================================
        self.video_grouper = VideoGrouper()

        # ====================================================
        # 8. TEMPORAL CLUSTERER
        # ====================================================
        self.temporal_clusterer = TemporalClusterer(
            max_gap_seconds=max_gap_seconds,
            max_cluster_duration=max_cluster_duration,
        )

        # ====================================================
        # 9. FUSION
        # ====================================================
        self.fusion_scorer = FusionScorer(
            best_weight=best_weight,
            second_weight=second_weight,
            coverage_weight=coverage_weight,
            object_boost_weight=object_boost_weight,
        )

        # ====================================================
        # 10. RANKER
        # ====================================================
        self.ranker = CandidateRanker(top_n=top_n)

        # ====================================================
        # READY
        # ====================================================
        print("\n" + "=" * 42)
        print("[INFO] RetrievalEngine READY")
        print(f"[INFO] Top-K/query : {self.top_k}")
        print(f"[INFO] Top-N video : {self.top_n}")
        print(f"[INFO] Object      : {self.object_enabled}")
        print("=" * 42 + "\n")

    # ========================================================
    # EXTRACT RETRIEVAL QUERIES
    # ========================================================

    def _extract_retrieval_queries(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Lấy expanded_queries từ Analyzer.

        Expected:
            analysis["retrieval"]["expanded_queries"]
        """
        if not isinstance(analysis, dict):
            raise TypeError("analysis phải là dict.")

        retrieval = analysis.get("retrieval")
        if not isinstance(retrieval, dict):
            raise ValueError("analysis không có retrieval hợp lệ.")

        queries = retrieval.get("expanded_queries", [])
        if not queries:
            raise ValueError("expanded_queries đang rỗng.")

        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise TypeError(f"Query {i} không phải dict.")
            if not query.get("text"):
                raise ValueError(f"Query {i} thiếu text.")

        return queries

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        analysis: Dict[str, Any],
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """Chạy toàn bộ retrieval pipeline.

        Input:
            analysis: output của Query Analyzer.
        Output:
            Top-N candidate videos.
        """
        # ====================================================
        # CONFIG OVERRIDE
        # ====================================================
        if top_k is None:
            top_k = self.top_k
        if top_n is None:
            top_n = self.top_n

        if top_k <= 0:
            raise ValueError("top_k phải > 0.")
        if top_n <= 0:
            raise ValueError("top_n phải > 0.")

        # ====================================================
        # STEP 1: EXTRACT QUERIES
        # ====================================================
        queries = self._extract_retrieval_queries(analysis)

        if debug:
            print("\n" + "=" * 42)
            print("RETRIEVAL START")
            print("=" * 42)
            print(f"\n[1] Retrieval queries: {len(queries)}")

            for i, query in enumerate(queries):
                print(f"    Q{i}: {query['text']}")
                print(
                    f"        type={query.get('type', 'unknown')} | "
                    f"weight={query.get('weight', 1.0)}"
                )

            entities = analysis.get("entities", [])
            print(f"\n    Analyzer entities: {entities}")

        # ====================================================
        # STEP 2: CLIP ENCODE
        # ====================================================
        encoded_queries = self.encoder.encode_retrieval_queries(queries)

        if debug:
            print(f"\n[2] Encoded {len(encoded_queries)} queries")
            print(
                f"    embedding_dim={encoded_queries[0]['embedding'].shape[0]}"
            )

        # ====================================================
        # STEP 3: FAISS SEARCH
        # ====================================================
        search_results = self.searcher.search_queries(
            encoded_queries=encoded_queries, top_k=top_k
        )

        if debug:
            total_search_hits = sum(
                len(result.get("hits", [])) for result in search_results
            )
            print(f"\n[3] FAISS hits: {total_search_hits}")

        # ====================================================
        # STEP 4: METADATA MAPPING
        # ====================================================
        mapped_results = self.mapper.map_search_results(search_results)

        if debug:
            print("\n[4] Metadata mapping done")

        # ====================================================
        # STEP 5: OBJECT ENRICHMENT
        # ====================================================
        if self.object_enabled:
            mapped_results = self.object_enricher.enrich_results(
                mapped_results=mapped_results, analysis=analysis
            )

            if debug:
                print("\n[5] Object enrichment done")
                object_hit_count = 0
                total_object_score = 0.0

                for result in mapped_results:
                    for hit in result.get("hits", []):
                        object_score = float(
                            hit.get("object_match_score", 0.0)
                        )
                        if object_score > 0:
                            object_hit_count += 1
                            total_object_score += object_score

                print(f"    hits with object evidence: {object_hit_count}")
                if object_hit_count > 0:
                    print(
                        "    mean object match score: "
                        f"{total_object_score / object_hit_count:.4f}"
                    )
        else:
            if debug:
                print("\n[5] Object enrichment skipped")

        # ====================================================
        # STEP 6: FLATTEN / UNION
        # ====================================================
        hit_pool = self.hit_pool_builder.flatten(mapped_results)

        if debug:
            print(f"\n[6] Raw hit pool: {len(hit_pool)}")

        # ====================================================
        # STEP 7: GROUP EXACT SAME KEYFRAME
        # ====================================================
        grouped_hits = self.hit_pool_builder.group_by_feature(hit_pool)

        if debug:
            print(f"\n[7] Unique keyframes: {len(grouped_hits)}")

        # ====================================================
        # STEP 8: GROUP BY VIDEO
        # ====================================================
        video_groups = self.video_grouper.group(grouped_hits)

        if debug:
            print(f"\n[8] Candidate videos: {len(video_groups)}")

        # ====================================================
        # STEP 9: TEMPORAL CLUSTERING
        # ====================================================
        clustered_videos = self.temporal_clusterer.cluster_all(video_groups)

        if debug:
            total_clusters = sum(
                len(video.get("temporal_clusters", []))
                for video in clustered_videos
            )
            print(f"\n[9] Temporal clusters: {total_clusters}")

        # ====================================================
        # STEP 10: SEMANTIC FUSION + OBJECT BOOST
        # ====================================================
        scored_videos = self.fusion_scorer.score_all(
            clustered_videos=clustered_videos, encoded_queries=encoded_queries
        )

        if debug:
            print("\n[10] Fusion scoring done")

        # ====================================================
        # STEP 11: FINAL VIDEO RANKING
        # ====================================================
        candidates = self.ranker.rank(
            scored_videos=scored_videos, top_n=top_n
        )

        # ====================================================
        # DEBUG FINAL RESULTS
        # ====================================================
        if debug:
            print(f"\n[11] Final candidates: {len(candidates)}")
            print("\n" + "=" * 42)
            print("TOP CANDIDATES")
            print("=" * 42)

            for candidate in candidates:
                rank = candidate.get("rank", 0)
                video_id = candidate.get("video_id")
                final_score = float(candidate.get("score", 0.0))
                semantic_score = float(
                    candidate.get("semantic_score", final_score)
                )
                object_score = float(candidate.get("object_score", 0.0))
                object_boost = float(candidate.get("object_boost", 0.0))
                boost_factor = float(candidate.get("boost_factor", 1.0))
                candidate_time = float(candidate.get("candidate_time", 0.0))
                candidate_frame = candidate.get("candidate_frame")

                print(f"\n#{rank:02d} {video_id}")
                print(f"    final_score    = {final_score:.6f}")
                print(f"    semantic_score = {semantic_score:.6f}")
                print(f"    object_score   = {object_score:.6f}")
                print(f"    object_boost   = {object_boost:.6f}")
                print(f"    boost_factor   = {boost_factor:.6f}")
                print(f"    time           = {candidate_time:.2f}s")
                print(f"    frame          = {candidate_frame}")

            print("\n" + "=" * 42)
            print("RETRIEVAL END")
            print("=" * 42)

        return candidates

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """Release resources có handle mở.

        Hiện tại chủ yếu dùng để đóng ObjectStore ZIP.
        """
        if self.object_store is not None:
            self.object_store.close()

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()