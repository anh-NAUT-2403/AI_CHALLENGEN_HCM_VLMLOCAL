# video_pipeline/video_cache.py

import os
import shutil
import zipfile

from typing import Dict, Any, List, Optional


class VideoCache:
    """
    Extract trực tiếp đúng video member
    từ ZIP trên Google Drive -> local Colab.

    Không copy toàn bộ ZIP.

    Ví dụ:

        Drive:
            Videos_L24_a.zip

        Member:
            video/L24_V012.mp4

        Output:
            /content/video_cache/videos/L24_V012.mp4
    """

    def __init__(
        self,
        cache_root: str = "/content/video_cache"
    ):

        self.cache_root = os.path.abspath(
            cache_root
        )

        self.video_cache_dir = os.path.join(
            self.cache_root,
            "videos"
        )

        os.makedirs(
            self.video_cache_dir,
            exist_ok=True
        )


    # ========================================================
    # LOCAL VIDEO PATH
    # ========================================================

    def get_local_video_path(
        self,
        video_id: str
    ) -> str:

        if not isinstance(
            video_id,
            str
        ):
            raise TypeError(
                "video_id phải là string."
            )

        video_id = video_id.strip()

        if not video_id:
            raise ValueError(
                "video_id đang rỗng."
            )

        return os.path.join(
            self.video_cache_dir,
            f"{video_id}.mp4"
        )


    # ========================================================
    # DIRECT EXTRACT
    # ========================================================

    def extract_video(
        self,
        video_id: str,
        source_zip_path: str,
        member_path: str,
        force_extract: bool = False
    ) -> str:
        """
        Mở ZIP trực tiếp từ Google Drive
        và stream đúng member về local.

        KHÔNG copy toàn bộ ZIP.
        """

        local_video_path = (
            self.get_local_video_path(
                video_id
            )
        )


        # ====================================================
        # VIDEO CACHE HIT
        # ====================================================

        if (
            os.path.isfile(
                local_video_path
            )
            and not force_extract
        ):

            print(
                f"[VideoCache] Cache hit: "
                f"{video_id}"
            )

            return local_video_path


        # ====================================================
        # SOURCE ZIP CHECK
        # ====================================================

        if not os.path.isfile(
            source_zip_path
        ):

            raise FileNotFoundError(
                f"Không tìm thấy ZIP: "
                f"{source_zip_path}"
            )


        print(
            f"[VideoCache] Direct extract:"
        )

        print(
            f"    ZIP   : {source_zip_path}"
        )

        print(
            f"    MEMBER: {member_path}"
        )

        print(
            f"    OUTPUT: {local_video_path}"
        )


        # ====================================================
        # DIRECT STREAM FROM DRIVE ZIP
        # ====================================================

        try:

            with zipfile.ZipFile(
                source_zip_path,
                "r"
            ) as zf:

                # getinfo nhanh hơn gọi namelist()
                # rồi search toàn bộ list.
                try:

                    member_info = zf.getinfo(
                        member_path
                    )

                except KeyError:

                    raise KeyError(
                        f"Không tìm thấy member "
                        f"{member_path} "
                        f"trong ZIP."
                    )


                print(
                    "[VideoCache] Member size: "
                    f"{member_info.file_size / 1024 / 1024:.2f} MB"
                )


                with zf.open(
                    member_info,
                    "r"
                ) as source:

                    with open(
                        local_video_path,
                        "wb"
                    ) as destination:

                        shutil.copyfileobj(
                            source,
                            destination,
                            length=1024 * 1024
                        )


        except zipfile.BadZipFile:

            raise RuntimeError(
                f"ZIP bị lỗi: "
                f"{source_zip_path}"
            )


        # ====================================================
        # VERIFY
        # ====================================================

        if not os.path.isfile(
            local_video_path
        ):

            raise RuntimeError(
                "Extract video thất bại."
            )


        file_size = os.path.getsize(
            local_video_path
        )


        if file_size <= 0:

            raise RuntimeError(
                "Video output có size = 0."
            )


        print(
            "[VideoCache] Done: "
            f"{file_size / 1024 / 1024:.2f} MB"
        )


        return local_video_path


    # ========================================================
    # CACHE LOCATION
    # ========================================================

    def cache_location(
        self,
        location: Dict[str, Any],
        force_extract: bool = False
    ) -> Dict[str, Any]:

        if not isinstance(
            location,
            dict
        ):

            raise TypeError(
                "location phải là dict."
            )


        video_id = location.get(
            "video_id"
        )

        zip_path = location.get(
            "zip_path"
        )

        member_path = location.get(
            "member_path"
        )


        if not video_id:

            raise ValueError(
                "location thiếu video_id."
            )


        if not zip_path:

            raise ValueError(
                "location thiếu zip_path."
            )


        if not member_path:

            raise ValueError(
                "location thiếu member_path."
            )


        local_video_path = (
            self.extract_video(
                video_id=video_id,
                source_zip_path=zip_path,
                member_path=member_path,
                force_extract=force_extract
            )
        )


        result = dict(
            location
        )


        result.update(
            {
                "local_video_path":
                    local_video_path,

                "cached":
                    True
            }
        )


        return result


    # ========================================================
    # CACHE RETRIEVAL CANDIDATE
    # ========================================================

    def cache_candidate(
        self,
        candidate: Dict[str, Any],
        force_extract: bool = False
    ) -> Dict[str, Any]:

        if not isinstance(
            candidate,
            dict
        ):

            raise TypeError(
                "candidate phải là dict."
            )


        video_id = candidate.get(
            "video_id"
        )


        if not video_id:

            raise ValueError(
                "candidate thiếu video_id."
            )


        if candidate.get(
            "video_found"
        ) is False:

            result = dict(
                candidate
            )

            result.update(
                {
                    "video_cached":
                        False,

                    "local_video_path":
                        None
                }
            )

            return result


        source_zip_path = candidate.get(
            "video_zip_path"
        )

        member_path = candidate.get(
            "video_member_path"
        )


        if not source_zip_path:

            raise ValueError(
                "candidate thiếu video_zip_path."
            )


        if not member_path:

            raise ValueError(
                "candidate thiếu video_member_path."
            )


        local_video_path = (
            self.extract_video(
                video_id=video_id,
                source_zip_path=source_zip_path,
                member_path=member_path,
                force_extract=force_extract
            )
        )


        result = dict(
            candidate
        )


        result.update(
            {
                "video_cached":
                    True,

                "local_video_path":
                    local_video_path
            }
        )


        return result


    # ========================================================
    # CACHE MULTIPLE
    # ========================================================

    def cache_candidates(
        self,
        candidates: List[
            Dict[str, Any]
        ],
        limit: Optional[int] = None
    ) -> List[
        Dict[str, Any]
    ]:

        if not isinstance(
            candidates,
            list
        ):

            raise TypeError(
                "candidates phải là list."
            )


        selected = candidates


        if limit is not None:

            if limit <= 0:

                raise ValueError(
                    "limit phải > 0."
                )

            selected = candidates[
                :limit
            ]


        results = []


        for i, candidate in enumerate(
            selected,
            start=1
        ):

            video_id = candidate.get(
                "video_id",
                "UNKNOWN"
            )


            print(
                f"\n[VideoCache] "
                f"{i}/{len(selected)} "
                f"{video_id}"
            )


            try:

                result = (
                    self.cache_candidate(
                        candidate
                    )
                )

            except Exception as error:

                result = dict(
                    candidate
                )

                result.update(
                    {
                        "video_cached":
                            False,

                        "local_video_path":
                            None,

                        "video_cache_error":
                            str(error)
                    }
                )


            results.append(
                result
            )


        return results


    # ========================================================
    # STATUS
    # ========================================================

    def is_video_cached(
        self,
        video_id: str
    ) -> bool:

        path = (
            self.get_local_video_path(
                video_id
            )
        )

        return os.path.isfile(
            path
        )


    # ========================================================
    # CLEAR
    # ========================================================

    def clear_video_cache(
        self
    ):

        if os.path.isdir(
            self.video_cache_dir
        ):

            shutil.rmtree(
                self.video_cache_dir
            )


        os.makedirs(
            self.video_cache_dir,
            exist_ok=True
        )