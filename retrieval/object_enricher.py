
from typing import List, Dict, Any


class ObjectEnricher:
    """
    Thêm object evidence vào từng FAISS hit.

    Sử dụng:
        ObjectStore
            để lấy object detections của keyframe

        ObjectMatcher
            để semantic-match Analyzer entities
            với detected object classes

    Module này KHÔNG thay đổi raw_score của CLIP.
    Nó chỉ thêm object evidence vào mỗi hit.
    """

    def __init__(
        self,
        object_store,
        object_matcher,
        detection_threshold: float = 0.3
    ):
        if object_store is None:
            raise ValueError(
                "object_store không được None."
            )

        if object_matcher is None:
            raise ValueError(
                "object_matcher không được None."
            )

        if not (
            0.0 <= detection_threshold <= 1.0
        ):
            raise ValueError(
                "detection_threshold phải trong [0, 1]."
            )

        self.object_store = object_store
        self.object_matcher = object_matcher

        self.detection_threshold = float(
            detection_threshold
        )


    # ========================================================
    # ENRICH ONE HIT
    # ========================================================

    def enrich_hit(
        self,
        hit: Dict[str, Any],
        query_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Thêm object information cho một keyframe hit.

        Input hit:

        {
            "feature_index": ...,
            "video_id": "L21_V001",
            "keyframe_id": 1,
            ...
        }

        Output:

        hit cũ
        +
        {
            "object_match_score": ...,
            "object_coverage": ...,
            "object_matched_count": ...,
            "object_matches": ...
        }
        """

        video_id = hit.get(
            "video_id"
        )

        keyframe_id = hit.get(
            "keyframe_id"
        )

        if not video_id:
            raise ValueError(
                "Hit không có video_id."
            )

        if keyframe_id is None:
            raise ValueError(
                "Hit không có keyframe_id."
            )


        # ----------------------------------------------------
        # Không có query entity
        # → không có object evidence
        # ----------------------------------------------------

        if not query_entities:

            result = dict(
                hit
            )

            result.update(
                {
                    "object_match_score": 0.0,
                    "object_coverage": 0.0,
                    "object_matched_count": 0,
                    "object_query_entity_count": 0,
                    "object_matches": []
                }
            )

            return result


        # ----------------------------------------------------
        # Load object detections
        # ----------------------------------------------------

        detections = (
            self.object_store
            .get_detections(
                video_id=video_id,
                keyframe_id=int(
                    keyframe_id
                ),
                score_threshold=(
                    self.detection_threshold
                )
            )
        )


        # ----------------------------------------------------
        # Không có object metadata
        # ----------------------------------------------------

        if not detections:

            result = dict(
                hit
            )

            result.update(
                {
                    "object_match_score": 0.0,
                    "object_coverage": 0.0,
                    "object_matched_count": 0,
                    "object_query_entity_count":
                        len(query_entities),
                    "object_matches": []
                }
            )

            return result


        # ----------------------------------------------------
        # Semantic object matching
        # ----------------------------------------------------

        match_result = (
            self.object_matcher.match(
                query_entities=query_entities,
                detections=detections
            )
        )


        result = dict(
            hit
        )

        result.update(
            {
                "object_match_score":
                    float(
                        match_result[
                            "object_match_score"
                        ]
                    ),

                "object_coverage":
                    float(
                        match_result[
                            "object_coverage"
                        ]
                    ),

                "object_matched_count":
                    int(
                        match_result[
                            "matched_count"
                        ]
                    ),

                "object_query_entity_count":
                    int(
                        match_result[
                            "query_entity_count"
                        ]
                    ),

                "object_matches":
                    match_result[
                        "matches"
                    ]
            }
        )

        return result


    # ========================================================
    # ENRICH ALL SEARCH RESULTS
    # ========================================================

    def enrich_results(
        self,
        mapped_results: List[
            Dict[str, Any]
        ],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Nhận output trực tiếp từ HitMapper.

        mapped_results:

        [
            {
                "query_id": 0,
                "text": "...",
                "weight": 1.0,
                "hits": [...]
            }
        ]

        analysis:
            output Query Analyzer.

        Sử dụng:
            analysis["entities"]
        """

        if not mapped_results:
            raise ValueError(
                "mapped_results đang rỗng."
            )

        query_entities = analysis.get(
            "entities",
            []
        )

        if not isinstance(
            query_entities,
            list
        ):
            raise TypeError(
                "analysis['entities'] phải là list."
            )


        enriched_results = []


        for result in mapped_results:

            enriched_hits = []


            for hit in result[
                "hits"
            ]:

                enriched_hit = (
                    self.enrich_hit(
                        hit=hit,
                        query_entities=(
                            query_entities
                        )
                    )
                )

                enriched_hits.append(
                    enriched_hit
                )


            new_result = dict(
                result
            )

            new_result[
                "hits"
            ] = enriched_hits

            enriched_results.append(
                new_result
            )


        return enriched_results
