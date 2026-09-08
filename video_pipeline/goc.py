import os
import shutil
import zipfile

from typing import Dict, Any, List, Optional


class VideoCache:
    """
    Quản lý video candidate trên local disk của Colab.

    Flow:

        Drive ZIP
            ↓
        copy ZIP về local một lần
            ↓
        extract đúng member cần dùng
            ↓
        local .mp4

    Ví dụ:

        Drive:
            Videos_L24_a.zip

        Local ZIP:
            /content/video_cache/zips/
            Videos_L24_a.zip

        Extract:
            video/L24_V012.mp4

        Local video:
            /content/video_cache/videos/
            L24_V012.mp4
    """

    def __init__(
        self,
        cache_root: str = "/content/video_cache",
        keep_local_zips: bool = True
    ):
        """
        Parameters
        ----------
        cache_root:
            Root local cache trên Colab.

        keep_local_zips:
            True:
                giữ ZIP local để candidate khác
                trong cùng ZIP dùng lại.

            False:
                có thể xóa ZIP sau khi extract.
                Không khuyên dùng ở giai đoạn hiện tại.
        """

        self.cache_root = os.path.abspath(
            cache_root
        )

        self.keep_local_zips = bool(
            keep_local_zips
        )


        # ====================================================
        # CACHE DIRECTORIES
        # ====================================================

        self.zip_cache_dir = os.path.join(
            self.cache_root,
            "zips"
        )

        self.video_cache_dir = os.path.join(
            self.cache_root,
            "videos"
        )


        os.makedirs(
            self.zip_cache_dir,
            exist_ok=True
        )

        os.makedirs(
            self.video_cache_dir,
            exist_ok=True
        )


    # ========================================================
    # LOCAL ZIP PATH
    # ========================================================

    def get_local_zip_path(
        self,
        source_zip_path: str
    ) -> str:
        """
        Chuyển:

            /drive/.../Videos_L24_a.zip

        thành:

            /content/video_cache/zips/
            Videos_L24_a.zip
        """

        if not isinstance(
            source_zip_path,
            str
        ):
            raise TypeError(
                "source_zip_path phải là string."
            )


        source_zip_path = (
            source_zip_path.strip()
        )


        if not source_zip_path:
            raise ValueError(
                "source_zip_path đang rỗng."
            )


        zip_name = os.path.basename(
            source_zip_path
        )


        return os.path.join(
            self.zip_cache_dir,
            zip_name
        )


    # ========================================================
    # ENSURE ZIP LOCAL
    # ========================================================

    def ensure_zip_local(
        self,
        source_zip_path: str,
        force_copy: bool = False
    ) -> str:
        """
        Đảm bảo ZIP đã tồn tại trên local disk.

        Nếu đã cache:
            reuse.

        Nếu chưa:
            copy từ Drive → /content.
        """

        if not os.path.isfile(
            source_zip_path
        ):
            raise FileNotFoundError(
                f"Không tìm thấy source ZIP: "
                f"{source_zip_path}"
            )


        local_zip_path = (
            self.get_local_zip_path(
                source_zip_path
            )
        )


        # ====================================================
        # CACHE HIT
        # ====================================================

        if (
            os.path.isfile(
                local_zip_path
            )
            and not force_copy
        ):

            return local_zip_path


        # ====================================================
        # COPY DRIVE -> LOCAL
        # ====================================================

        print(
            f"[VideoCache] Copy ZIP:"
        )

        print(
            f"    FROM: {source_zip_path}"
        )

        print(
            f"    TO  : {local_zip_path}"
        )


        shutil.copy2(
            source_zip_path,
            local_zip_path
        )


        # ====================================================
        # VERIFY
        # ====================================================

        if not os.path.isfile(
            local_zip_path
        ):
            raise RuntimeError(
                "Copy ZIP thất bại."
            )


        return local_zip_path


    # ========================================================
    # LOCAL VIDEO PATH
    # ========================================================

    def get_local_video_path(
        self,
        video_id: str
    ) -> str:
        """
        Ví dụ:

            L24_V012

        ->

            /content/video_cache/videos/
            L24_V012.mp4
        """

        if not isinstance(
            video_id,
            str
        ):
            raise TypeError(
                "video_id phải là string."
            )


        video_id = (
            video_id.strip()
        )


        if not video_id:
            raise ValueError(
                "video_id đang rỗng."
            )


        return os.path.join(
            self.video_cache_dir,
            f"{video_id}.mp4"
        )


    # ========================================================
    # EXTRACT MEMBER
    # ========================================================

    def extract_video(
        self,
        video_id: str,
        source_zip_path: str,
        member_path: str,
        force_extract: bool = False
    ) -> str:
        """
        Copy ZIP về local nếu cần,
        sau đó chỉ extract đúng member video.

        Không unzip toàn archive.
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

            return local_video_path


        # ====================================================
        # ENSURE ZIP LOCAL
        # ====================================================

        local_zip_path = (
            self.ensure_zip_local(
                source_zip_path
            )
        )


        # ====================================================
        # OPEN LOCAL ZIP
        # ====================================================

        try:

            with zipfile.ZipFile(
                local_zip_path,
                "r"
            ) as zf:

                # --------------------------------------------
                # MEMBER CHECK
                # --------------------------------------------

                if member_path not in zf.namelist():

                    raise KeyError(
                        f"Không tìm thấy member "
                        f"'{member_path}' "
                        f"trong {local_zip_path}"
                    )


                print(
                    f"[VideoCache] Extract:"
                )

                print(
                    f"    {member_path}"
                )

                print(
                    f"    -> {local_video_path}"
                )


                # =================================================
                # Stream đúng member ra destination.
                #
                # Không dùng extract() vì extract sẽ giữ folder:
                #
                # video/L24_V012.mp4
                #
                # Ta muốn file cuối trực tiếp:
                #
                # videos/L24_V012.mp4
                # =================================================

                with zf.open(
                    member_path,
                    "r"
                ) as source:

                    with open(
                        local_video_path,
                        "wb"
                    ) as destination:

                        shutil.copyfileobj(
                            source,
                            destination
                        )


        except zipfile.BadZipFile:

            raise RuntimeError(
                f"ZIP bị lỗi: "
                f"{local_zip_path}"
            )


        # ====================================================
        # VERIFY OUTPUT
        # ====================================================

        if not os.path.isfile(
            local_video_path
        ):
            raise RuntimeError(
                "Extract video thất bại."
            )


        if os.path.getsize(
            local_video_path
        ) <= 0:

            raise RuntimeError(
                "Video extract ra có size = 0."
            )


        # ====================================================
        # OPTIONAL REMOVE ZIP
        # ====================================================

        if not self.keep_local_zips:

            try:

                os.remove(
                    local_zip_path
                )

            except OSError:

                pass


        return local_video_path


    # ========================================================
    # CACHE LOCATION RESULT
    # ========================================================

    def cache_location(
        self,
        location: Dict[str, Any],
        force_extract: bool = False
    ) -> Dict[str, Any]:
        """
        Nhận trực tiếp output từ:

            VideoLocator.locate()

        Expected:

        {
            "video_id":
                "L24_V012",

            "zip_path":
                "...Videos_L24_a.zip",

            "member_path":
                "video/L24_V012.mp4"
        }
        """

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

                force_extract=(
                    force_extract
                )
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
        """
        Nhận output từ:

            VideoLocator.locate_candidates()

        Expected candidate:

        {
            "video_id": ...,

            "video_found": True,

            "video_zip_path": ...,

            "video_member_path": ...,

            ...
        }

        Trả candidate cũ +
        local_video_path.
        """

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

                source_zip_path=(
                    source_zip_path
                ),

                member_path=(
                    member_path
                ),

                force_extract=(
                    force_extract
                )
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
    # CACHE MANY CANDIDATES
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
        """
        Cache nhiều Retrieval candidates.

        Có thể giới hạn:

            limit=5

        để chỉ đưa Top-5 video về local.
        """

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
                f"\n"
                f"[VideoCache] "
                f"Candidate {i}/{len(selected)}: "
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
    # CACHE STATUS
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
    # CLEAR VIDEOS
    # ========================================================

    def clear_video_cache(
        self
    ):
        """
        Xóa extracted MP4 nhưng giữ ZIP local.
        """

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


    # ========================================================
    # CLEAR ZIP CACHE
    # ========================================================

    def clear_zip_cache(
        self
    ):
        """
        Xóa ZIP đã copy về local.
        """

        if os.path.isdir(
            self.zip_cache_dir
        ):

            shutil.rmtree(
                self.zip_cache_dir
            )


        os.makedirs(
            self.zip_cache_dir,
            exist_ok=True
        )


    # ========================================================
    # CLEAR ALL
    # ========================================================

    def clear_all(
        self
    ):
        """
        Xóa toàn bộ local video cache.
        """

        if os.path.isdir(
            self.cache_root
        ):

            shutil.rmtree(
                self.cache_root
            )


        os.makedirs(
            self.zip_cache_dir,
            exist_ok=True
        )

        os.makedirs(
            self.video_cache_dir,
            exist_ok=True
        )
