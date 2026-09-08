
from typing import List, Dict, Any


class VideoGrouper:
    """
    Gom các unique keyframe hits theo video_id.
    """

    def group(
        self,
        grouped_hits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not grouped_hits:
            raise ValueError(
                "grouped_hits đang rỗng."
            )

        videos = {}

        for hit in grouped_hits:

            video_id = hit["video_id"]

            if video_id not in videos:

                videos[video_id] = {
                    "video_id": video_id,
                    "hits": []
                }

            videos[video_id]["hits"].append(
                hit
            )

        results = []

        for video_id, video in videos.items():

            # Sort hit theo thời gian
            video["hits"].sort(
                key=lambda x: x["pts_time"]
            )

            # ================================================
            # DEBUG STATISTICS
            # ================================================

            query_ids = set()

            for hit in video["hits"]:

                for query_hit in hit["query_hits"]:

                    query_ids.add(
                        query_hit["query_id"]
                    )

            video["unique_query_count"] = len(
                query_ids
            )

            video["unique_hit_count"] = len(
                video["hits"]
            )

            # ================================================

            results.append(
                video
            )

        # Chỉ sort tạm để dễ debug.
        # Đây KHÔNG phải final video ranking.
        results.sort(
            key=lambda x: x[
                "unique_hit_count"
            ],
            reverse=True
        )

        return results
