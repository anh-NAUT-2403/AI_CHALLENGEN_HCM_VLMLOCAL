from typing import List, Dict, Any


class TemporalClusterer:
    """
    Gom các keyframe hit gần nhau về thời gian
    thành temporal windows.

    Điều kiện để một hit được thêm vào cluster hiện tại:

    1. Khoảng cách với hit trước <= max_gap_seconds
    2. Tổng độ dài cluster <= max_cluster_duration
    """

    def __init__(
        self,
        max_gap_seconds: float = 10.0,
        max_cluster_duration: float = 30.0
    ):
        if max_gap_seconds <= 0:
            raise ValueError(
                "max_gap_seconds phải > 0."
            )

        if max_cluster_duration <= 0:
            raise ValueError(
                "max_cluster_duration phải > 0."
            )

        self.max_gap_seconds = float(
            max_gap_seconds
        )

        self.max_cluster_duration = float(
            max_cluster_duration
        )


    # ========================================================
    # CLUSTER ONE VIDEO
    # ========================================================

    def cluster_video(
        self,
        video: Dict[str, Any]
    ) -> Dict[str, Any]:

        hits = video.get(
            "hits",
            []
        )

        if not hits:
            return {
                "video_id": video["video_id"],
                "temporal_clusters": []
            }

        # Sort theo timestamp
        hits = sorted(
            hits,
            key=lambda x: x["pts_time"]
        )

        clusters = []

        current_cluster = [
            hits[0]
        ]

        # ----------------------------------------------------
        # BUILD CLUSTERS
        # ----------------------------------------------------

        for hit in hits[1:]:

            previous_hit = (
                current_cluster[-1]
            )

            cluster_start = (
                current_cluster[0]
            )

            gap = (
                hit["pts_time"]
                - previous_hit["pts_time"]
            )

            cluster_duration = (
                hit["pts_time"]
                - cluster_start["pts_time"]
            )

            if (
                gap <= self.max_gap_seconds
                and
                cluster_duration
                <= self.max_cluster_duration
            ):

                current_cluster.append(
                    hit
                )

            else:

                clusters.append(
                    current_cluster
                )

                current_cluster = [
                    hit
                ]

        # Cluster cuối
        clusters.append(
            current_cluster
        )

        # ----------------------------------------------------
        # NORMALIZE OUTPUT
        # ----------------------------------------------------

        temporal_clusters = []

        for cluster_id, cluster_hits in enumerate(
            clusters,
            start=1
        ):

            start_time = (
                cluster_hits[0][
                    "pts_time"
                ]
            )

            end_time = (
                cluster_hits[-1][
                    "pts_time"
                ]
            )

            start_frame = (
                cluster_hits[0][
                    "frame_id"
                ]
            )

            end_frame = (
                cluster_hits[-1][
                    "frame_id"
                ]
            )

            temporal_clusters.append(
                {
                    "cluster_id":
                        cluster_id,

                    "start_time":
                        float(start_time),

                    "end_time":
                        float(end_time),

                    "duration":
                        float(
                            end_time
                            - start_time
                        ),

                    "start_frame":
                        int(start_frame),

                    "end_frame":
                        int(end_frame),

                    "hit_count":
                        len(cluster_hits),

                    "hits":
                        cluster_hits
                }
            )

        return {
            "video_id":
                video["video_id"],

            "temporal_clusters":
                temporal_clusters
        }


    # ========================================================
    # CLUSTER ALL VIDEOS
    # ========================================================

    def cluster_all(
        self,
        video_groups: List[
            Dict[str, Any]
        ]
    ) -> List[Dict[str, Any]]:

        if not video_groups:
            raise ValueError(
                "video_groups đang rỗng."
            )

        results = []

        for video in video_groups:

            clustered_video = (
                self.cluster_video(
                    video
                )
            )

            results.append(
                clustered_video
            )

        return results