from typing import List, Dict, Any

import numpy as np


class ObjectMatcher:
    """
    Match entities từ Query Analyzer với object detector
    bằng CLIP semantic similarity.

    Quy trình:

        exact string match
            ↓ nếu không có
        CLIP semantic similarity
            ↓
        chọn detected entity semantic gần nhất
            ↓
        threshold
            ↓
        kết hợp detection confidence
    """

    def __init__(
        self,
        encoder,
        min_detection_score: float = 0.3,
        semantic_threshold: float = 0.83
    ):
        if not (
            0.0 <= min_detection_score <= 1.0
        ):
            raise ValueError(
                "min_detection_score phải nằm trong [0, 1]."
            )

        if not (
            0.0 <= semantic_threshold <= 1.0
        ):
            raise ValueError(
                "semantic_threshold phải nằm trong [0, 1]."
            )

        self.encoder = encoder

        self.min_detection_score = float(
            min_detection_score
        )

        self.semantic_threshold = float(
            semantic_threshold
        )

        # Cache CLIP embedding của entity
        self.embedding_cache = {}


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_text(
        self,
        text: str
    ) -> str:

        if not isinstance(text, str):
            return ""

        return (
            text
            .strip()
            .lower()
        )


    # ========================================================
    # EMBEDDING
    # ========================================================

    def get_embedding(
        self,
        text: str
    ) -> np.ndarray:

        text = self.normalize_text(
            text
        )

        if not text:

            raise ValueError(
                "Không thể encode entity rỗng."
            )

        if text in self.embedding_cache:

            return self.embedding_cache[
                text
            ]

        embedding = (
            self.encoder.encode(
                text
            )
        )

        self.embedding_cache[
            text
        ] = embedding

        return embedding


    # ========================================================
    # SIMILARITY
    # ========================================================

    def similarity(
        self,
        text_a: str,
        text_b: str
    ) -> float:

        text_a = self.normalize_text(
            text_a
        )

        text_b = self.normalize_text(
            text_b
        )

        if not text_a or not text_b:
            return 0.0

        # Exact match
        if text_a == text_b:
            return 1.0

        embedding_a = (
            self.get_embedding(
                text_a
            )
        )

        embedding_b = (
            self.get_embedding(
                text_b
            )
        )

        # embeddings đã L2 normalize
        score = np.dot(
            embedding_a,
            embedding_b
        )

        return float(
            score
        )


    # ========================================================
    # PREPARE DETECTIONS
    # ========================================================

    def prepare_detections(
        self,
        detections: List[
            Dict[str, Any]
        ]
    ) -> List[Dict[str, Any]]:
        """
        Lọc detection confidence thấp.

        Nếu một class xuất hiện nhiều lần,
        chỉ giữ detection có confidence cao nhất.
        """

        best_by_entity = {}


        for detection in detections:

            entity = self.normalize_text(
                detection.get(
                    "entity",
                    ""
                )
            )

            score = float(
                detection.get(
                    "score",
                    0.0
                )
            )

            if not entity:
                continue

            if (
                score
                < self.min_detection_score
            ):
                continue


            previous = best_by_entity.get(
                entity
            )


            if (
                previous is None
                or
                score
                >
                previous[
                    "detection_score"
                ]
            ):

                best_by_entity[
                    entity
                ] = {
                    "entity":
                        entity,

                    "detection_score":
                        score,

                    "box":
                        detection.get(
                            "box"
                        )
                }


        return list(
            best_by_entity.values()
        )


    # ========================================================
    # MATCH ONE ENTITY
    # ========================================================

    def match_entity(
        self,
        query_entity: str,
        detections: List[
            Dict[str, Any]
        ]
    ) -> Dict[str, Any]:

        query_entity = (
            self.normalize_text(
                query_entity
            )
        )

        prepared = (
            self.prepare_detections(
                detections
            )
        )


        if not prepared:

            return {
                "query_entity":
                    query_entity,

                "detected_entity":
                    None,

                "semantic_similarity":
                    0.0,

                "detection_score":
                    0.0,

                "matched":
                    False,

                "combined_score":
                    0.0,

                "box":
                    None
            }


        # ====================================================
        # 1. EXACT MATCH FIRST
        # ====================================================

        for detection in prepared:

            if (
                detection["entity"]
                == query_entity
            ):

                detection_score = float(
                    detection[
                        "detection_score"
                    ]
                )

                return {
                    "query_entity":
                        query_entity,

                    "detected_entity":
                        detection["entity"],

                    "semantic_similarity":
                        1.0,

                    "detection_score":
                        detection_score,

                    "matched":
                        True,

                    "combined_score":
                        detection_score,

                    "box":
                        detection.get(
                            "box"
                        )
                }


        # ====================================================
        # 2. SEMANTIC SEARCH
        # ====================================================

        best = None


        for detection in prepared:

            detected_entity = (
                detection["entity"]
            )

            semantic_score = (
                self.similarity(
                    query_entity,
                    detected_entity
                )
            )


            candidate = {
                "query_entity":
                    query_entity,

                "detected_entity":
                    detected_entity,

                "semantic_similarity":
                    float(
                        semantic_score
                    ),

                "detection_score":
                    float(
                        detection[
                            "detection_score"
                        ]
                    ),

                "box":
                    detection.get(
                        "box"
                    )
            }


            # IMPORTANT:
            # chọn object theo SEMANTIC similarity,
            # không chọn theo detection confidence.
            if (
                best is None
                or
                candidate[
                    "semantic_similarity"
                ]
                >
                best[
                    "semantic_similarity"
                ]
            ):

                best = candidate


        # ====================================================
        # 3. THRESHOLD
        # ====================================================

        matched = (
            best[
                "semantic_similarity"
            ]
            >=
            self.semantic_threshold
        )

        best[
            "matched"
        ] = matched


        # ====================================================
        # 4. OBJECT EVIDENCE SCORE
        # ====================================================

        if matched:

            best[
                "combined_score"
            ] = (
                best[
                    "semantic_similarity"
                ]
                *
                best[
                    "detection_score"
                ]
            )

        else:

            best[
                "combined_score"
            ] = 0.0


        return best


    # ========================================================
    # MATCH ALL ENTITIES
    # ========================================================

    def match(
        self,
        query_entities: List[str],
        detections: List[
            Dict[str, Any]
        ]
    ) -> Dict[str, Any]:

        # Normalize + unique
        normalized_entities = []

        seen = set()


        for entity in query_entities:

            entity = self.normalize_text(
                entity
            )

            if not entity:
                continue

            if entity in seen:
                continue

            seen.add(
                entity
            )

            normalized_entities.append(
                entity
            )


        if not normalized_entities:

            return {
                "object_match_score":
                    0.0,

                "object_coverage":
                    0.0,

                "matched_count":
                    0,

                "query_entity_count":
                    0,

                "matches":
                    []
            }


        matches = []

        total_score = 0.0

        matched_count = 0


        for query_entity in normalized_entities:

            result = (
                self.match_entity(
                    query_entity,
                    detections
                )
            )

            matches.append(
                result
            )


            if result["matched"]:

                matched_count += 1

                total_score += (
                    result[
                        "combined_score"
                    ]
                )


        # Missing entity đóng góp 0
        object_match_score = (
            total_score
            /
            len(
                normalized_entities
            )
        )


        object_coverage = (
            matched_count
            /
            len(
                normalized_entities
            )
        )


        return {
            "object_match_score":
                float(
                    object_match_score
                ),

            "object_coverage":
                float(
                    object_coverage
                ),

            "matched_count":
                matched_count,

            "query_entity_count":
                len(
                    normalized_entities
                ),

            "matches":
                matches
        }