from typing import List, Dict, Any

import faiss
import numpy as np


class FaissSearcher:
    """
    Search query embeddings trên FAISS index.

    Input:
        encoded queries từ CLIPTextEncoder

    Output:
        Top-K feature indices + raw FAISS scores
        cho từng query.

    Chưa map metadata ở module này.
    """

    def __init__(
        self,
        index
    ):
        if index is None:
            raise ValueError(
                "FAISS index không được None."
            )

        self.index = index

        self.dimension = index.d
        self.metric_type = getattr(
            index,
            "metric_type",
            None
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_embedding(
        self,
        embedding: np.ndarray
    ):
        """
        Kiểm tra một query embedding trước khi search.
        """

        if not isinstance(
            embedding,
            np.ndarray
        ):
            raise TypeError(
                "Embedding phải là numpy.ndarray."
            )

        if embedding.ndim != 1:
            raise ValueError(
                f"Embedding phải có shape (D,), "
                f"nhưng nhận {embedding.shape}."
            )

        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension = "
                f"{embedding.shape[0]}, "
                f"nhưng FAISS dimension = "
                f"{self.dimension}."
            )


    # ========================================================
    # SINGLE QUERY SEARCH
    # ========================================================

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Search một query embedding.

        Output:
        {
            "indices": ndarray shape (K,),
            "scores": ndarray shape (K,)
        }
        """

        self._validate_embedding(
            embedding
        )

        if top_k <= 0:
            raise ValueError(
                "top_k phải > 0."
            )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        # FAISS yêu cầu float32
        query = np.asarray(
            embedding,
            dtype=np.float32
        )

        # Từ (512,) -> (1, 512)
        query = query.reshape(
            1,
            -1
        )

        scores, indices = self.index.search(
            query,
            top_k
        )

        return {
            "indices": indices[0],
            "scores": scores[0]
        }


    # ========================================================
    # MULTI-QUERY SEARCH
    # ========================================================

    def search_queries(
        self,
        encoded_queries: List[Dict[str, Any]],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search từng retrieval query độc lập.

        Input example:

        [
            {
                "query_id": 0,
                "text": "...",
                "type": "global",
                "weight": 1.0,
                "embedding": ndarray(...)
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
                "hits": [
                    {
                        "rank": 1,
                        "feature_index": 123,
                        "raw_score": ...
                    },
                    ...
                ]
            }
        ]
        """

        if not encoded_queries:
            raise ValueError(
                "encoded_queries đang rỗng."
            )

        all_results = []

        for query in encoded_queries:

            embedding = query.get(
                "embedding"
            )

            if embedding is None:
                raise ValueError(
                    f"Query "
                    f"{query.get('query_id')} "
                    "không có embedding."
                )

            result = self.search(
                embedding=embedding,
                top_k=top_k
            )

            hits = []

            for rank, (
                feature_index,
                raw_score
            ) in enumerate(
                zip(
                    result["indices"],
                    result["scores"]
                ),
                start=1
            ):

                # FAISS đôi khi có thể trả -1
                # nếu không đủ result.
                if feature_index < 0:
                    continue

                hits.append(
                    {
                        "rank": rank,
                        "feature_index": int(
                            feature_index
                        ),
                        "raw_score": float(
                            raw_score
                        )
                    }
                )

            all_results.append(
                {
                    "query_id": query[
                        "query_id"
                    ],
                    "text": query[
                        "text"
                    ],
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
                    "hits": hits
                }
            )

        return all_results