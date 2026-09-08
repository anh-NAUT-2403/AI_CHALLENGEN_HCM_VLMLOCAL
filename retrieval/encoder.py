from typing import List, Dict, Any, Union

import numpy as np
import torch
import clip


class CLIPTextEncoder:
    """
    Encode retrieval query thành CLIP text embedding.

    Model mặc định:
        OpenAI CLIP ViT-B/32

    Output:
        numpy.ndarray float32

    Shape:
        single query: (512,)
        multiple queries: (N, 512)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.model_name = model_name

        print(
            f"[INFO] Loading CLIP text encoder: "
            f"{self.model_name}"
        )

        print(
            f"[INFO] Device: {self.device}"
        )

        self.model, _ = clip.load(
            self.model_name,
            device=self.device
        )

        self.model.eval()

        print(
            "[INFO] CLIP text encoder loaded."
        )


    # ========================================================
    # SINGLE QUERY
    # ========================================================

    def encode(
        self,
        text: str
    ) -> np.ndarray:
        """
        Encode một query.

        Output:
            shape = (512,)
            dtype = float32
            L2 normalized
        """

        if not isinstance(text, str):
            raise TypeError(
                "text phải là string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Query text đang rỗng."
            )

        tokens = clip.tokenize(
            [text],
            truncate=True
        ).to(self.device)

        with torch.no_grad():

            features = self.model.encode_text(
                tokens
            )

            features = features.float()

            # L2 normalize
            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        embedding = (
            features[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding


    # ========================================================
    # MULTIPLE QUERIES
    # ========================================================

    def encode_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Encode nhiều query cùng lúc.

        Input:
            [
                "a man speaking",
                "outdoor press conference",
                ...
            ]

        Output:
            numpy.ndarray
            shape = (N, 512)
        """

        if not texts:
            raise ValueError(
                "Danh sách queries đang rỗng."
            )

        clean_texts = []

        for i, text in enumerate(texts):

            if not isinstance(text, str):
                raise TypeError(
                    f"Query {i} không phải string."
                )

            text = text.strip()

            if not text:
                raise ValueError(
                    f"Query {i} đang rỗng."
                )

            clean_texts.append(text)

        tokens = clip.tokenize(
            clean_texts,
            truncate=True
        ).to(self.device)

        with torch.no_grad():

            features = self.model.encode_text(
                tokens
            )

            features = features.float()

            # Normalize từng embedding
            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        embeddings = (
            features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embeddings


    # ========================================================
    # ANALYZER QUERIES
    # ========================================================

    def encode_retrieval_queries(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Nhận trực tiếp expanded_queries từ Query Analyzer.

        Input example:

        [
            {
                "text": "...",
                "type": "global",
                "weight": 1.0
            },
            ...
        ]

        Output:

        [
            {
                "query_id": 0,
                "text": "...",
                "type": "global",
                "weight": 1.0,
                "embedding": np.ndarray(...)
            },
            ...
        ]
        """

        if not queries:
            raise ValueError(
                "retrieval queries đang rỗng."
            )

        texts = [
            query["text"]
            for query in queries
        ]

        embeddings = self.encode_batch(
            texts
        )

        results = []

        for i, (
            query,
            embedding
        ) in enumerate(
            zip(
                queries,
                embeddings
            )
        ):

            results.append(
                {
                    "query_id": i,
                    "text": query["text"],
                    "type": query.get(
                        "type",
                        "unknown"
                    ),
                    "weight": float(
                        query.get(
                            "weight",
                            1.0
                        )
                    ),
                    "embedding": embedding
                }
            )

        return results