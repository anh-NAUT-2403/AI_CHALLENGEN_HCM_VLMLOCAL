# video_pipeline/vlm_reranker.py

from typing import (
    Dict,
    Any,
    List
)


class VLMReranker:
    """
    Continuous VLM reranking.

    VLM confidence được hiểu là:

        visual/query match score

    chứ KHÔNG phải:
        confidence rằng verified=True/False.


    Nếu match_score == threshold:
        factor = 1.0

    Nếu match_score > threshold:
        boost tăng dần.

    Nếu match_score < threshold:
        penalty tăng dần.
    """


    def __init__(
        self,
        positive_boost: float = 0.50,
        reject_penalty: float = 0.80,
        minimum_factor: float = 0.10,
        match_threshold: float = 0.50
    ):

        if positive_boost < 0:

            raise ValueError(
                "positive_boost phải >= 0."
            )


        if not (
            0.0
            <= reject_penalty
            <= 1.0
        ):

            raise ValueError(
                "reject_penalty phải trong [0,1]."
            )


        if not (
            0.0
            <= minimum_factor
            <= 1.0
        ):

            raise ValueError(
                "minimum_factor phải trong [0,1]."
            )


        if not (
            0.0
            <
            match_threshold
            <
            1.0
        ):

            raise ValueError(
                "match_threshold phải trong (0,1)."
            )


        self.positive_boost = float(
            positive_boost
        )

        self.reject_penalty = float(
            reject_penalty
        )

        self.minimum_factor = float(
            minimum_factor
        )

        self.match_threshold = float(
            match_threshold
        )


    # ========================================================
    # CLAMP
    # ========================================================

    @staticmethod
    def _clamp01(
        value
    ) -> float:

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0


        return min(
            max(
                value,
                0.0
            ),
            1.0
        )


    # ========================================================
    # SCORE ONE
    # ========================================================

    def score_candidate(
        self,
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # RETRIEVAL SCORE
        # ----------------------------------------------------

        retrieval_score = float(

            candidate.get(
                "retrieval_score",

                candidate.get(
                    "score",
                    0.0
                )
            )
        )


        # ----------------------------------------------------
        # VLM STATE
        # ----------------------------------------------------

        vlm_processed = bool(
            candidate.get(
                "vlm_processed",
                False
            )
        )


        vlm_error = candidate.get(
            "vlm_error"
        )


        match_score = (
            self._clamp01(

                candidate.get(
                    "vlm_confidence",
                    0.0
                )
            )
        )


        threshold = (
            self.match_threshold
        )


        # ====================================================
        # NOT PROCESSED / ERROR
        # ====================================================

        if (
            not vlm_processed
            or
            vlm_error
        ):

            factor = 1.0

            status = (
                "unverified"
            )

            positive_strength = 0.0

            mismatch_strength = 0.0


        # ====================================================
        # MATCHED
        # ====================================================

        elif (
            match_score
            >=
            threshold
        ):

            # -----------------------------------------------
            # Normalize range:
            #
            # threshold -> 0
            # 1.0       -> 1
            # -----------------------------------------------

            positive_strength = (

                (
                    match_score
                    -
                    threshold
                )

                /

                (
                    1.0
                    -
                    threshold
                )
            )


            mismatch_strength = 0.0


            factor = (

                1.0

                +

                self.positive_boost
                *
                positive_strength
            )


            status = (
                "verified"
            )


        # ====================================================
        # WEAK / REJECTED
        # ====================================================

        else:

            # -----------------------------------------------
            # Normalize range:
            #
            # threshold -> 0
            # 0.0       -> 1
            # -----------------------------------------------

            mismatch_strength = (

                (
                    threshold
                    -
                    match_score
                )

                /

                threshold
            )


            positive_strength = 0.0


            factor = (

                1.0

                -

                self.reject_penalty
                *
                mismatch_strength
            )


            factor = max(
                factor,
                self.minimum_factor
            )


            status = (
                "rejected"
            )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        final_score = (

            retrieval_score
            *
            factor
        )


        result = dict(
            candidate
        )


        result.update(
            {

                # Original retrieval score
                "retrieval_score":
                    float(
                        retrieval_score
                    ),


                # Continuous VLM visual match score
                "vlm_match_score":
                    float(
                        match_score
                    ),


                # Threshold used
                "vlm_match_threshold":
                    float(
                        threshold
                    ),


                # Debug
                "vlm_positive_strength":
                    float(
                        positive_strength
                    ),

                "vlm_mismatch_strength":
                    float(
                        mismatch_strength
                    ),


                # Status
                "vlm_rerank_status":
                    status,


                # Multiplicative factor
                "vlm_rerank_factor":
                    float(
                        factor
                    ),


                # Final score
                "final_score":
                    float(
                        final_score
                    ),


                # Main score after reranking
                "score":
                    float(
                        final_score
                    )
            }
        )


        return result


    # ========================================================
    # RANK ALL
    # ========================================================

    def rank(
        self,
        candidates: List[
            Dict[str, Any]
        ]
    ) -> List[
        Dict[str, Any]
    ]:

        scored = []


        for candidate in candidates:

            scored.append(

                self.score_candidate(
                    candidate
                )
            )


        scored.sort(

            key=lambda x:
                x.get(
                    "final_score",
                    0.0
                ),

            reverse=True
        )


        for new_rank, candidate in enumerate(
            scored,
            start=1
        ):

            # Giữ ranking retrieval ban đầu.
            candidate[
                "retrieval_rank"
            ] = candidate.get(
                "rank"
            )


            candidate[
                "rank"
            ] = new_rank


        return scored