import pickle
from pathlib import Path
from typing import Any

import faiss


class RetrievalDataStore:
    """
    Load và giữ các dữ liệu dùng chung cho Retrieval.

    Bao gồm:
    - FAISS index
    - metadata mapping

    Các dữ liệu này chỉ nên load 1 lần khi khởi động hệ thống.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.metadata = None


    # ========================================================
    # LOAD FAISS INDEX
    # ========================================================

    def load_index(self):
        """
        Load FAISS index từ disk.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index: {self.index_path}"
            )

        print(f"[INFO] Loading FAISS index: {self.index_path}")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        print("[INFO] FAISS index loaded.")
        print(f"       ntotal = {self.index.ntotal}")
        print(f"       dim    = {self.index.d}")

        return self.index


    # ========================================================
    # LOAD METADATA
    # ========================================================

    def load_metadata(self):
        """
        Load metadata pickle.
        """

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy metadata: {self.metadata_path}"
            )

        print(f"[INFO] Loading metadata: {self.metadata_path}")

        with open(
            self.metadata_path,
            "rb"
        ) as f:
            self.metadata = pickle.load(f)

        print("[INFO] Metadata loaded.")
        print(
            f"       type = {type(self.metadata)}"
        )

        try:
            print(
                f"       size = {len(self.metadata)}"
            )
        except TypeError:
            pass

        return self.metadata


    # ========================================================
    # LOAD ALL
    # ========================================================

    def load(self):
        """
        Load toàn bộ dữ liệu Retrieval.
        """

        self.load_index()
        self.load_metadata()

        self._basic_validation()

        return self


    # ========================================================
    # VALIDATION
    # ========================================================

    def _basic_validation(self):
        """
        Kiểm tra sơ bộ index và metadata.
        """

        if self.index is None:
            raise RuntimeError(
                "FAISS index chưa được load."
            )

        if self.metadata is None:
            raise RuntimeError(
                "Metadata chưa được load."
            )

        print("\n[INFO] Basic validation")

        try:
            metadata_size = len(self.metadata)

            print(
                f"       FAISS vectors : {self.index.ntotal}"
            )
            print(
                f"       Metadata items: {metadata_size}"
            )

            if self.index.ntotal != metadata_size:
                print(
                    "[WARNING] Số vector FAISS "
                    "không bằng số metadata item."
                )

        except TypeError:
            print(
                "[WARNING] Metadata không hỗ trợ len(). "
                "Cần kiểm tra cấu trúc metadata."
            )


    # ========================================================
    # INSPECTION
    # ========================================================

    def inspect_metadata(
        self,
        n: int = 5
    ):
        """
        In vài metadata item đầu tiên để xem cấu trúc thật.
        """

        if self.metadata is None:
            raise RuntimeError(
                "Metadata chưa được load."
            )

        print("\n========== METADATA INSPECTION ==========")

        print(
            "Metadata type:",
            type(self.metadata)
        )

        # ----------------------------------------------------
        # LIST
        # ----------------------------------------------------

        if isinstance(
            self.metadata,
            (list, tuple)
        ):

            print(
                f"Number of items: {len(self.metadata)}"
            )

            for i, item in enumerate(
                self.metadata[:n]
            ):
                print(f"\n--- Item {i} ---")
                print(
                    "type:",
                    type(item)
                )
                print(item)

        # ----------------------------------------------------
        # DICT
        # ----------------------------------------------------

        elif isinstance(
            self.metadata,
            dict
        ):

            print(
                f"Number of keys: {len(self.metadata)}"
            )

            keys = list(
                self.metadata.keys()
            )

            for key in keys[:n]:

                print(f"\n--- Key {key} ---")

                value = self.metadata[key]

                print(
                    "value type:",
                    type(value)
                )

                print(value)

        # ----------------------------------------------------
        # OTHER
        # ----------------------------------------------------

        else:

            print(
                "Unknown metadata structure."
            )

            print(
                self.metadata
            )