
from typing import List, Dict, Any


class HitMapper:
    """
    Map kết quả FAISS sang metadata video/keyframe.

    Mapping:
        FAISS feature_index
                ↓
        metadata[feature_index]

    Metadata format:
        {
            "video_id": ...,
            "keyframe_id": ...,
            "frame_id": ...,
            "pts_time": ...,
            "fps": ...
        }
    """

    def __init__(self, metadata):
        if metadata is None:
            raise ValueError("Metadata không được None.")

        if not isinstance(metadata, (list, tuple)):
            raise TypeError(
                "Metadata hiện tại phải là list hoặc tuple."
            )

        self.metadata = metadata

    def map_feature(
        self,
        feature_index: int
    ) -> Dict[str, Any]:
        """
        Map một FAISS feature index sang metadata.
        """

        if not isinstance(feature_index, int):
            raise TypeError(
                "feature_index phải là int."
            )

        if (
            feature_index < 0
            or feature_index >= len(self.metadata)
        ):
            raise IndexError(
                f"feature_index={feature_index} "
                "nằm ngoài phạm vi metadata."
            )

        item = self.metadata[feature_index]

        if not isinstance(item, dict):
            raise TypeError(
                f"metadata[{feature_index}] không phải dict."
            )

        required_fields = [
            "video_id",
            "keyframe_id",
            "frame_id",
            "pts_time",
            "fps"
        ]

        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"metadata[{feature_index}] "
                    f"thiếu field '{field}'."
                )

        return {
            "video_id": item["video_id"],
            "keyframe_id": int(item["keyframe_id"]),
            "frame_id": int(item["frame_id"]),
            "pts_time": float(item["pts_time"]),
            "fps": float(item["fps"])
        }

    def map_search_results(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map toàn bộ kết quả từ FaissSearcher.
        """

        if not search_results:
            raise ValueError(
                "search_results đang rỗng."
            )

        mapped_results = []

        for result in search_results:

            mapped_hits = []

            for hit in result["hits"]:

                feature_index = hit["feature_index"]

                metadata = self.map_feature(
                    feature_index
                )

                mapped_hits.append(
                    {
                        "rank": hit["rank"],
                        "feature_index": feature_index,
                        "raw_score": float(
                            hit["raw_score"]
                        ),

                        "video_id": metadata["video_id"],
                        "keyframe_id": metadata["keyframe_id"],
                        "frame_id": metadata["frame_id"],
                        "pts_time": metadata["pts_time"],
                        "fps": metadata["fps"]
                    }
                )

            mapped_results.append(
                {
                    "query_id": result["query_id"],
                    "text": result["text"],
                    "type": result["type"],
                    "weight": result["weight"],
                    "hits": mapped_hits
                }
            )

        return mapped_results
