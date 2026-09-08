
import os
import glob
import zipfile

from typing import Dict, List, Optional, Any


class VideoLocator:

    def __init__(
        self,
        videos_root: str,
        zip_pattern: str = "Videos_*.zip",
        auto_build: bool = True
    ):

        self.videos_root = videos_root
        self.zip_pattern = zip_pattern

        if not os.path.isdir(
            self.videos_root
        ):
            raise FileNotFoundError(
                f"Videos root không tồn tại: "
                f"{self.videos_root}"
            )

        self.video_index: Dict[
            str,
            Dict[str, str]
        ] = {}

        self.zip_files: List[str] = []

        if auto_build:
            self.build_index()


    # ========================================================
    # FIND ZIP FILES
    # ========================================================

    def find_zip_files(
        self
    ) -> List[str]:

        pattern = os.path.join(
            self.videos_root,
            self.zip_pattern
        )

        return sorted(
            glob.glob(
                pattern
            )
        )


    # ========================================================
    # MEMBER -> VIDEO ID
    # ========================================================

    def _member_to_video_id(
        self,
        member_path: str
    ) -> Optional[str]:

        if not isinstance(
            member_path,
            str
        ):
            return None

        member_path = (
            member_path.strip()
        )

        if not member_path:
            return None

        if member_path.endswith("/"):
            return None

        filename = os.path.basename(
            member_path
        )

        root, ext = os.path.splitext(
            filename
        )

        if ext.lower() != ".mp4":
            return None

        if not root:
            return None

        return root


    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build_index(
        self
    ) -> Dict[
        str,
        Dict[str, str]
    ]:

        self.video_index = {}

        self.zip_files = (
            self.find_zip_files()
        )

        if not self.zip_files:
            raise FileNotFoundError(
                "Không tìm thấy Videos_*.zip trong: "
                f"{self.videos_root}"
            )

        print(
            "\n=========================================="
        )
        print(
            "BUILDING VIDEO INDEX"
        )
        print(
            "=========================================="
        )

        print(
            f"Video ZIPs found: "
            f"{len(self.zip_files)}"
        )

        duplicate_count = 0

        for i, zip_path in enumerate(
            self.zip_files,
            start=1
        ):

            zip_name = os.path.basename(
                zip_path
            )

            print(
                f"\n[{i}/{len(self.zip_files)}] "
                f"{zip_name}"
            )

            try:

                with zipfile.ZipFile(
                    zip_path,
                    "r"
                ) as zf:

                    video_count = 0

                    for member_path in (
                        zf.namelist()
                    ):

                        video_id = (
                            self._member_to_video_id(
                                member_path
                            )
                        )

                        if video_id is None:
                            continue

                        video_count += 1

                        if (
                            video_id
                            in self.video_index
                        ):

                            duplicate_count += 1

                            print(
                                f"[WARNING] Duplicate: "
                                f"{video_id}"
                            )

                            continue

                        self.video_index[
                            video_id
                        ] = {
                            "video_id":
                                video_id,

                            "zip_path":
                                zip_path,

                            "member_path":
                                member_path
                        }

                    print(
                        f"    Videos indexed: "
                        f"{video_count}"
                    )

            except zipfile.BadZipFile:

                print(
                    f"[ERROR] Bad ZIP: "
                    f"{zip_path}"
                )

        print(
            "\n=========================================="
        )
        print(
            "VIDEO INDEX READY"
        )
        print(
            "=========================================="
        )

        print(
            f"Total videos indexed: "
            f"{len(self.video_index)}"
        )

        print(
            f"Duplicate video IDs: "
            f"{duplicate_count}"
        )

        return self.video_index


    # ========================================================
    # LOCATE ONE
    # ========================================================

    def locate(
        self,
        video_id: str,
        raise_if_missing: bool = True
    ) -> Optional[
        Dict[str, str]
    ]:

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

        result = (
            self.video_index.get(
                video_id
            )
        )

        if result is None:

            if raise_if_missing:
                raise KeyError(
                    f"Không tìm thấy video_id: "
                    f"{video_id}"
                )

            return None

        return dict(
            result
        )


    # ========================================================
    # LOCATE MANY
    # ========================================================

    def locate_many(
        self,
        video_ids: List[str],
        ignore_missing: bool = False
    ) -> List[
        Dict[str, Any]
    ]:

        results = []

        for video_id in video_ids:

            location = self.locate(
                video_id,
                raise_if_missing=False
            )

            if location is None:

                if ignore_missing:
                    continue

                results.append(
                    {
                        "video_id":
                            video_id,

                        "found":
                            False,

                        "zip_path":
                            None,

                        "member_path":
                            None
                    }
                )

            else:

                location["found"] = True

                results.append(
                    location
                )

        return results


    # ========================================================
    # LOCATE RETRIEVAL CANDIDATES
    # ========================================================

    def locate_candidates(
        self,
        candidates: List[
            Dict[str, Any]
        ],
        ignore_missing: bool = False
    ) -> List[
        Dict[str, Any]
    ]:

        results = []

        for candidate in candidates:

            video_id = candidate.get(
                "video_id"
            )

            if not video_id:
                continue

            location = self.locate(
                video_id,
                raise_if_missing=False
            )

            result = dict(
                candidate
            )

            if location is None:

                if ignore_missing:
                    continue

                result.update(
                    {
                        "video_found":
                            False,

                        "video_zip_path":
                            None,

                        "video_member_path":
                            None
                    }
                )

            else:

                result.update(
                    {
                        "video_found":
                            True,

                        "video_zip_path":
                            location[
                                "zip_path"
                            ],

                        "video_member_path":
                            location[
                                "member_path"
                            ]
                    }
                )

            results.append(
                result
            )

        return results


    # ========================================================
    # INFO
    # ========================================================

    def __len__(
        self
    ) -> int:

        return len(
            self.video_index
        )


    def get_stats(
        self
    ) -> Dict[str, int]:

        return {
            "zip_count":
                len(
                    self.zip_files
                ),

            "video_count":
                len(
                    self.video_index
                )
        }
