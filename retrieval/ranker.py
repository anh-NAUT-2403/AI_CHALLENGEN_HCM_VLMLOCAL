from typing import List, Dict, Any


class CandidateRanker:
    """
    Rank candidate video dựa trên temporal cluster tốt nhất.

    Quy tắc:

        video_score =
            max(cluster["score"])

    Trong đó cluster["score"] đã bao gồm:
        semantic score
        + object boost

    Output cuối cùng là Top-N candidate videos.
    """

    def __init__(
        self,
        top_n: int = 20
    ):
        if top_n <= 0:
            raise ValueError(
                "top_n phải > 0."
            )

        self.top_n = int(
            top_n
        )


    # ========================================================
    # RANK ONE VIDEO
    # ========================================================

    def build_video_candidate(
        self,
        video: Dict[str, Any]
    ) -> Dict[str, Any]:

        video_id = video[
            "video_id"
        ]

        clusters = video.get(
            "temporal_clusters",
            []
        )


        if not clusters:
            return None


        # ====================================================
        # BEST CLUSTER
        # ====================================================

        # Dùng FINAL SCORE.
        #
        # score đã bao gồm:
        # semantic_score * object boost factor
        best_cluster = max(
            clusters,
            key=lambda x: x[
                "score"
            ]
        )


        # ====================================================
        # BUILD FINAL CANDIDATE
        # ====================================================

        candidate = {

            # ------------------------------------------------
            # ID
            # ------------------------------------------------

            "video_id":
                video_id,


            # ------------------------------------------------
            # FINAL SCORE
            # ------------------------------------------------

            "score":
                float(
                    best_cluster[
                        "score"
                    ]
                ),


            # ------------------------------------------------
            # SEMANTIC SCORE
            # ------------------------------------------------

            "semantic_score":
                float(
                    best_cluster.get(
                        "semantic_score",
                        0.0
                    )
                ),


            # ------------------------------------------------
            # OBJECT SCORE / BOOST
            # ------------------------------------------------

            "object_score":
                float(
                    best_cluster.get(
                        "object_score",
                        0.0
                    )
                ),

            "best_object_score":
                float(
                    best_cluster.get(
                        "best_object_score",
                        0.0
                    )
                ),

            "mean_object_score":
                float(
                    best_cluster.get(
                        "mean_object_score",
                        0.0
                    )
                ),

            "best_object_coverage":
                float(
                    best_cluster.get(
                        "best_object_coverage",
                        0.0
                    )
                ),

            "object_boost":
                float(
                    best_cluster.get(
                        "object_boost",
                        0.0
                    )
                ),

            "object_boost_weight":
                float(
                    best_cluster.get(
                        "object_boost_weight",
                        0.0
                    )
                ),

            "boost_factor":
                float(
                    best_cluster.get(
                        "boost_factor",
                        1.0
                    )
                ),


            # ------------------------------------------------
            # REPRESENTATIVE FRAME
            # ------------------------------------------------

            "candidate_frame":
                int(
                    best_cluster[
                        "candidate_frame"
                    ]
                ),

            "candidate_time":
                float(
                    best_cluster[
                        "candidate_time"
                    ]
                ),


            # ------------------------------------------------
            # TEMPORAL WINDOW
            # ------------------------------------------------

            "temporal_window": {

                "start_frame":
                    int(
                        best_cluster[
                            "start_frame"
                        ]
                    ),

                "end_frame":
                    int(
                        best_cluster[
                            "end_frame"
                        ]
                    ),

                "start_time":
                    float(
                        best_cluster[
                            "start_time"
                        ]
                    ),

                "end_time":
                    float(
                        best_cluster[
                            "end_time"
                        ]
                    )
            },


            # ------------------------------------------------
            # CLUSTER INFO
            # ------------------------------------------------

            "cluster_id":
                int(
                    best_cluster[
                        "cluster_id"
                    ]
                ),

            "hit_count":
                int(
                    best_cluster[
                        "hit_count"
                    ]
                ),


            # ------------------------------------------------
            # QUERY FUSION INFO
            # ------------------------------------------------

            "matched_query_count":
                int(
                    best_cluster[
                        "matched_query_count"
                    ]
                ),

            "query_coverage":
                float(
                    best_cluster[
                        "query_coverage"
                    ]
                ),

            "best_score":
                float(
                    best_cluster[
                        "best_score"
                    ]
                ),

            "second_best_score":
                float(
                    best_cluster[
                        "second_best_score"
                    ]
                ),

            "query_best_scores":
                best_cluster[
                    "query_best_scores"
                ],


            # ------------------------------------------------
            # RAW HITS
            # ------------------------------------------------

            "hits":
                best_cluster[
                    "hits"
                ]
        }


        return candidate


    # ========================================================
    # RANK ALL VIDEOS
    # ========================================================

    def rank(
        self,
        scored_videos: List[
            Dict[str, Any]
        ],
        top_n: int = None
    ) -> List[Dict[str, Any]]:

        if not scored_videos:

            raise ValueError(
                "scored_videos đang rỗng."
            )


        if top_n is None:

            top_n = (
                self.top_n
            )


        if top_n <= 0:

            raise ValueError(
                "top_n phải > 0."
            )


        candidates = []


        # ====================================================
        # BUILD VIDEO CANDIDATES
        # ====================================================

        for video in scored_videos:

            candidate = (
                self.build_video_candidate(
                    video
                )
            )


            if candidate is not None:

                candidates.append(
                    candidate
                )


        # ====================================================
        # FINAL VIDEO RANKING
        # ====================================================

        candidates.sort(

            key=lambda x:
                x[
                    "score"
                ],

            reverse=True
        )


        # ====================================================
        # ADD RANK
        # ====================================================

        for rank, candidate in enumerate(
            candidates,
            start=1
        ):

            candidate[
                "rank"
            ] = rank


        # ====================================================
        # TOP-N
        # ====================================================

        return candidates[
            :top_n
        ]