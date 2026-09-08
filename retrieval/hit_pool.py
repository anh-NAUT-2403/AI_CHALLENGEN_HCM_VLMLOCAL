
from typing import List, Dict, Any


class HitPoolBuilder:
    """
    Chuyển kết quả search theo từng query
    thành unified hit pool.

    Đồng thời có thể group các hit cùng feature_index
    để phát hiện nhiều query cùng tìm thấy một keyframe.
    """

    def flatten(
        self,
        mapped_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not mapped_results:
            raise ValueError(
                "mapped_results đang rỗng."
            )

        hit_pool = []

        for result in mapped_results:

            query_id = result["query_id"]
            query_text = result["text"]
            query_type = result.get(
                "type",
                "unknown"
            )
            query_weight = float(
                result.get(
                    "weight",
                    1.0
                )
            )

            for hit in result["hits"]:

                hit_pool.append(
                {
                    "query_id":
                        query_id,

                    "query_text":
                        query_text,

                    "query_type":
                        query_type,

                    "query_weight":
                        query_weight,

                    "rank":
                        hit["rank"],

                    "feature_index":
                        hit["feature_index"],

                    "raw_score":
                        float(
                            hit["raw_score"]
                        ),

                    "video_id":
                        hit["video_id"],

                    "keyframe_id":
                        hit["keyframe_id"],

                    "frame_id":
                        hit["frame_id"],

                    "pts_time":
                        float(
                            hit["pts_time"]
                        ),

                    "fps":
                        float(
                            hit["fps"]
                        ),

                    # ================================================
                    # OBJECT EVIDENCE
                    # ================================================

                    "object_match_score":
                        float(
                            hit.get(
                                "object_match_score",
                                0.0
                            )
                        ),

                    "object_coverage":
                        float(
                            hit.get(
                                "object_coverage",
                                0.0
                            )
                        ),

                    "object_matched_count":
                        int(
                            hit.get(
                                "object_matched_count",
                                0
                            )
                        ),

                    "object_query_entity_count":
                        int(
                            hit.get(
                                "object_query_entity_count",
                                0
                            )
                        ),

                    "object_matches":
                        hit.get(
                            "object_matches",
                            []
                        )
                }
    )

        return hit_pool


    def group_by_feature(
        self,
        hit_pool: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        grouped = {}

        for hit in hit_pool:

            feature_index = hit[
                "feature_index"
            ]

            if feature_index not in grouped:

                grouped[
                    feature_index
                ] = {
                    "feature_index":
            feature_index,

        "video_id":
            hit["video_id"],

        "keyframe_id":
            hit["keyframe_id"],

        "frame_id":
            hit["frame_id"],

        "pts_time":
            hit["pts_time"],

        "fps":
            hit["fps"],

        # OBJECT
        "object_match_score":
            hit.get(
                "object_match_score",
                0.0
            ),

        "object_coverage":
            hit.get(
                "object_coverage",
                0.0
            ),

        "object_matched_count":
            hit.get(
                "object_matched_count",
                0
            ),

        "object_query_entity_count":
            hit.get(
                "object_query_entity_count",
                0
            ),

        "object_matches":
            hit.get(
                "object_matches",
                []
            ),

        "query_hits": []
                }

            grouped[
                feature_index
            ]["query_hits"].append(
                {
                    "query_id": hit["query_id"],
                    "query_text": hit["query_text"],
                    "query_type": hit["query_type"],
                    "query_weight": hit["query_weight"],
                    "rank": hit["rank"],
                    "raw_score": hit["raw_score"]
                }
            )

        results = list(
            grouped.values()
        )

        results.sort(
            key=lambda x: max(
                q["raw_score"]
                for q in x["query_hits"]
            ),
            reverse=True
        )

        return results
