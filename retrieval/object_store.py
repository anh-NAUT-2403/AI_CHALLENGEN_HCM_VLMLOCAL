
import json
import zipfile

from typing import Dict, Any, Optional


class ObjectStore:
    """
    Đọc object detection metadata trực tiếp từ
    objects-aic25-b1.zip.

    Expected structure:

        objects/
            L21_V001/
                001.json
                002.json
                ...
            L21_V002/
                001.json
                ...

    Mapping:

        (video_id, keyframe_id)
            ↓
        objects/{video_id}/{keyframe_id:03d}.json
    """

    def __init__(
        self,
        object_zip_path: str
    ):
        self.object_zip_path = (
            object_zip_path
        )

        print(
            "[INFO] Opening object metadata ZIP:"
        )

        print(
            f"       {self.object_zip_path}"
        )

        self.zip_file = zipfile.ZipFile(
            self.object_zip_path,
            "r"
        )

        # Dùng set để lookup filename O(1)
        self.file_names = set(
            self.zip_file.namelist()
        )

        print(
            "[INFO] Object metadata ZIP ready."
        )

        print(
            f"       files = "
            f"{len(self.file_names)}"
        )


    # ========================================================
    # BUILD PATH
    # ========================================================

    def build_object_path(
        self,
        video_id: str,
        keyframe_id: int
    ) -> str:
        """
        Ví dụ:

            video_id = L21_V001
            keyframe_id = 1

        ->

            objects/L21_V001/001.json
        """

        if not video_id:
            raise ValueError(
                "video_id đang rỗng."
            )

        if keyframe_id <= 0:
            raise ValueError(
                "keyframe_id phải >= 1."
            )

        return (
            f"objects/"
            f"{video_id}/"
            f"{keyframe_id:03d}.json"
        )


    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        video_id: str,
        keyframe_id: int
    ) -> bool:
        """
        Kiểm tra object JSON có tồn tại hay không.
        """

        path = self.build_object_path(
            video_id,
            keyframe_id
        )

        return (
            path in self.file_names
        )


    # ========================================================
    # GET RAW OBJECT DATA
    # ========================================================

    def get(
        self,
        video_id: str,
        keyframe_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Đọc JSON object detection của một keyframe.

        Returns:
            dict nếu tồn tại
            None nếu không tồn tại
        """

        path = self.build_object_path(
            video_id,
            keyframe_id
        )

        if path not in self.file_names:
            return None

        with self.zip_file.open(
            path,
            "r"
        ) as f:

            data = json.load(
                f
            )

        return data


    # ========================================================
    # GET DETECTIONS
    # ========================================================

    def get_detections(
        self,
        video_id: str,
        keyframe_id: int,
        score_threshold: float = 0.3
    ):
        """
        Trả detections đã được chuẩn hóa và lọc threshold.

        Output:

        [
            {
                "entity": "Car",
                "score": 0.82,
                "box": [...]
            },
            ...
        ]
        """

        data = self.get(
            video_id,
            keyframe_id
        )

        if data is None:
            return []

        scores = data.get(
            "detection_scores",
            []
        )

        entities = data.get(
            "detection_class_entities",
            []
        )

        boxes = data.get(
            "detection_boxes",
            []
        )

        detections = []

        for i, (
            score,
            entity
        ) in enumerate(
            zip(
                scores,
                entities
            )
        ):

            score = float(
                score
            )

            if score < score_threshold:
                continue

            box = None

            if i < len(boxes):

                box = [
                    float(x)
                    for x in boxes[i]
                ]

            detections.append(
                {
                    "entity":
                        entity,

                    "score":
                        score,

                    "box":
                        box
                }
            )

        return detections


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Đóng ZIP khi không còn sử dụng.
        """

        if self.zip_file is not None:

            self.zip_file.close()

            self.zip_file = None


    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.close()
