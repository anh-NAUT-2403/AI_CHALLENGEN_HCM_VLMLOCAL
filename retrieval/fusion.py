# retrieval/fusion.py

from typing import List, Dict, Any


class FusionScorer:
    """
    Tính fusion score cho từng temporal cluster.

    Semantic score:

        semantic_score =
            best_weight * best_query_score
            +
            second_weight * second_best_query_score
            +
            coverage_weight * query_coverage

    Object chỉ đóng vai trò BOOST:

        object_score =
            0.7 * best_object_score
            +
            0.3 * mean_object_score

        object_boost =
            object_score
            *
            best_object_coverage

        final_score =
            semantic_score
            *
            (
                1
                +
                object_boost_weight
                *
                object_boost
            )

    Object evidence không làm giảm semantic score.
    """

    def __init__(
        self,
        best_weight: float = 0.6,
        second_weight: float = 0.3,
        coverage_weight: float = 0.1,

        object_boost_weight: float = 0.2
    ):

        # ====================================================
        # VALIDATE QUERY FUSION WEIGHTS
        # ====================================================

        query_total = (
            best_weight
            + second_weight
            + coverage_weight
        )

        if abs(
            query_total - 1.0
        ) > 1e-6:

            raise ValueError(
                "best_weight + second_weight + "
                "coverage_weight phải bằng 1.0."
            )


        if object_boost_weight < 0:

            raise ValueError(
                "object_boost_weight phải >= 0."
            )


        self.best_weight = float(
            best_weight
        )

        self.second_weight = float(
            second_weight
        )

        self.coverage_weight = float(
            coverage_weight
        )

        self.object_boost_weight = float(
            object_boost_weight
        )


    # ========================================================
    # TOTAL QUERY WEIGHT
    # ========================================================

    def get_total_query_weight(
        self,
        encoded_queries: List[
            Dict[str, Any]
        ]
    ) -> float:
        """
        Tổng weight của tất cả retrieval queries.
        """

        if not encoded_queries:

            raise ValueError(
                "encoded_queries đang rỗng."
            )


        total = sum(

            float(
                query.get(
                    "weight",
                    1.0
                )
            )

            for query
            in encoded_queries
        )


        if total <= 0:

            raise ValueError(
                "Tổng query weight phải > 0."
            )


        return total


    # ========================================================
    # GET BEST HIT FOR EACH QUERY
    # ========================================================

    def get_query_best_scores(
        self,
        cluster: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Với mỗi retrieval query,
        chỉ lấy hit CLIP tốt nhất trong cluster.

        Query score:

            weighted_score =
                raw_score * query_weight
        """

        best_by_query = {}


        for hit in cluster.get(
            "hits",
            []
        ):

            for query_hit in hit.get(
                "query_hits",
                []
            ):

                query_id = (
                    query_hit[
                        "query_id"
                    ]
                )


                raw_score = float(
                    query_hit[
                        "raw_score"
                    ]
                )


                query_weight = float(
                    query_hit.get(
                        "query_weight",
                        1.0
                    )
                )


                weighted_score = (
                    raw_score
                    *
                    query_weight
                )


                candidate = {

                    "query_id":
                        query_id,

                    "query_weight":
                        query_weight,

                    "raw_score":
                        raw_score,

                    "weighted_score":
                        weighted_score,

                    "feature_index":
                        hit[
                            "feature_index"
                        ],

                    "frame_id":
                        hit[
                            "frame_id"
                        ],

                    "pts_time":
                        hit[
                            "pts_time"
                        ]
                }


                previous = (
                    best_by_query.get(
                        query_id
                    )
                )


                if (
                    previous is None
                    or
                    raw_score
                    >
                    previous[
                        "raw_score"
                    ]
                ):

                    best_by_query[
                        query_id
                    ] = candidate


        return best_by_query


    # ========================================================
    # OBJECT AGGREGATION
    # ========================================================

    def get_cluster_object_score(
        self,
        cluster: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Aggregate object evidence trong một temporal cluster.

        best_object_score:
            frame có object evidence tốt nhất.

        mean_object_score:
            mức object evidence trung bình trong cluster.

        best_object_coverage:
            coverage tốt nhất của query entities.

        object_score:
            0.7 * best
            +
            0.3 * mean
        """

        hits = cluster.get(
            "hits",
            []
        )


        if not hits:

            return {
                "object_score":
                    0.0,

                "best_object_score":
                    0.0,

                "mean_object_score":
                    0.0,

                "best_object_coverage":
                    0.0,

                "object_boost":
                    0.0
            }


        object_scores = [

            float(
                hit.get(
                    "object_match_score",
                    0.0
                )
            )

            for hit
            in hits
        ]


        object_coverages = [

            float(
                hit.get(
                    "object_coverage",
                    0.0
                )
            )

            for hit
            in hits
        ]


        best_object_score = max(
            object_scores,
            default=0.0
        )


        mean_object_score = (

            sum(
                object_scores
            )

            / len(
                object_scores
            )

            if object_scores
            else 0.0
        )


        best_object_coverage = max(
            object_coverages,
            default=0.0
        )


        object_score = (

            0.7
            *
            best_object_score

            +

            0.3
            *
            mean_object_score
        )


        object_boost = (

            object_score
            *
            best_object_coverage
        )


        return {

            "object_score":
                float(
                    object_score
                ),

            "best_object_score":
                float(
                    best_object_score
                ),

            "mean_object_score":
                float(
                    mean_object_score
                ),

            "best_object_coverage":
                float(
                    best_object_coverage
                ),

            "object_boost":
                float(
                    object_boost
                )
        }


    # ========================================================
    # SCORE ONE CLUSTER
    # ========================================================

    def score_cluster(
        self,
        cluster: Dict[str, Any],
        total_query_weight: float
    ) -> Dict[str, Any]:
        """
        Tính semantic score rồi dùng object evidence để boost.
        """

        # ====================================================
        # QUERY EVIDENCE
        # ====================================================

        query_best = (
            self.get_query_best_scores(
                cluster
            )
        )


        if not query_best:

            raise ValueError(
                "Cluster không có query evidence."
            )


        query_scores = list(
            query_best.values()
        )


        query_scores.sort(

            key=lambda x:
                x[
                    "weighted_score"
                ],

            reverse=True
        )


        # ====================================================
        # BEST / SECOND BEST
        # ====================================================

        best_score = float(
            query_scores[0][
                "weighted_score"
            ]
        )


        if len(
            query_scores
        ) >= 2:

            second_best_score = float(
                query_scores[1][
                    "weighted_score"
                ]
            )

        else:

            second_best_score = 0.0


        # ====================================================
        # QUERY COVERAGE
        # ====================================================

        matched_weight = sum(

            item[
                "query_weight"
            ]

            for item
            in query_scores
        )


        query_coverage = (

            matched_weight
            /
            total_query_weight
        )


        query_coverage = min(
            max(
                query_coverage,
                0.0
            ),
            1.0
        )


        # ====================================================
        # SEMANTIC SCORE
        # ====================================================

        semantic_score = (

            self.best_weight
            *
            best_score

            +

            self.second_weight
            *
            second_best_score

            +

            self.coverage_weight
            *
            query_coverage
        )


        # ====================================================
        # OBJECT EVIDENCE
        # ====================================================

        object_result = (
            self.get_cluster_object_score(
                cluster
            )
        )


        object_score = (
            object_result[
                "object_score"
            ]
        )


        object_boost = (
            object_result[
                "object_boost"
            ]
        )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        boost_factor = (

            1.0

            +

            self.object_boost_weight
            *
            object_boost
        )


        final_score = (

            semantic_score
            *
            boost_factor
        )


        # ====================================================
        # REPRESENTATIVE FRAME
        # ====================================================

        candidate_frame = int(
            query_scores[0][
                "frame_id"
            ]
        )


        candidate_time = float(
            query_scores[0][
                "pts_time"
            ]
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        scored_cluster = dict(
            cluster
        )


        scored_cluster.update(
            {

                # Semantic details
                "best_score":
                    float(
                        best_score
                    ),

                "second_best_score":
                    float(
                        second_best_score
                    ),

                "query_coverage":
                    float(
                        query_coverage
                    ),

                "matched_query_count":
                    len(
                        query_scores
                    ),

                "query_best_scores":
                    query_scores,


                # Object details
                "object_score":
                    float(
                        object_score
                    ),

                "best_object_score":
                    object_result[
                        "best_object_score"
                    ],

                "mean_object_score":
                    object_result[
                        "mean_object_score"
                    ],

                "best_object_coverage":
                    object_result[
                        "best_object_coverage"
                    ],

                "object_boost":
                    float(
                        object_boost
                    ),

                "object_boost_weight":
                    float(
                        self.object_boost_weight
                    ),

                "boost_factor":
                    float(
                        boost_factor
                    ),


                # Scores
                "semantic_score":
                    float(
                        semantic_score
                    ),

                "score":
                    float(
                        final_score
                    ),


                # Candidate position
                "candidate_frame":
                    candidate_frame,

                "candidate_time":
                    candidate_time
            }
        )


        return scored_cluster


    # ========================================================
    # SCORE ALL
    # ========================================================

    def score_all(
        self,
        clustered_videos: List[
            Dict[str, Any]
        ],
        encoded_queries: List[
            Dict[str, Any]
        ]
    ) -> List[Dict[str, Any]]:
        """
        Score toàn bộ temporal clusters.
        """

        if not clustered_videos:

            raise ValueError(
                "clustered_videos đang rỗng."
            )


        total_query_weight = (
            self.get_total_query_weight(
                encoded_queries
            )
        )


        results = []


        for video in clustered_videos:

            scored_clusters = []


            for cluster in video.get(
                "temporal_clusters",
                []
            ):

                scored = (
                    self.score_cluster(
                        cluster=cluster,
                        total_query_weight=(
                            total_query_weight
                        )
                    )
                )


                scored_clusters.append(
                    scored
                )


            scored_clusters.sort(

                key=lambda x:
                    x["score"],

                reverse=True
            )


            results.append(
                {

                    "video_id":
                        video[
                            "video_id"
                        ],

                    "temporal_clusters":
                        scored_clusters
                }
            )


        return results