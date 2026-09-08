import numpy as np


class TrakeEventRetriever:

    def __init__(
        self,
        store,
        encoder,
        top_k=200
    ):

        self.store = store
        self.encoder = encoder
        self.top_k = int(top_k)


    # ========================================================
    # EVENT TEXT
    # ========================================================

    @staticmethod
    def _get_event_text(event):

        for key in (
            "description",
            "action",
            "text"
        ):

            value = event.get(key)

            if value:
                return str(value)

        raise ValueError(
            f"Event thiếu description/action/text: {event}"
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        analysis,
        debug=False
    ):

        events = analysis.get(
            "events",
            []
        )

        if not events:

            raise ValueError(
                "TRAKE analysis không có events."
            )


        event_texts = [

            self._get_event_text(
                event
            )

            for event
            in events
        ]


        if debug:

            print(
                "\n=========================================="
            )

            print(
                "TRAKE EVENT RETRIEVAL"
            )

            print(
                "=========================================="
            )

            for i, text in enumerate(
                event_texts,
                start=1
            ):

                print(
                    f"E{i}: {text}"
                )


        # ====================================================
        # ENCODE ALL EVENTS
        # ====================================================

        embeddings = (
            self.encoder.encode_batch(
                event_texts
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim == 1:

            embeddings = embeddings[
                None,
                :
            ]

        embeddings = np.ascontiguousarray(
            embeddings
        )


        # ====================================================
        # FAISS
        # ====================================================

        scores, indices = (
            self.store.index.search(
                embeddings,
                self.top_k
            )
        )


        results = []


        # ====================================================
        # MAP METADATA
        # ====================================================

        for event_index, event in enumerate(
            events
        ):

            event_id = event.get(
                "event_id",
                event_index + 1
            )

            hits = []


            for rank, (
                score,
                feature_index
            ) in enumerate(

                zip(
                    scores[event_index],
                    indices[event_index]
                ),

                start=1
            ):

                feature_index = int(
                    feature_index
                )

                if feature_index < 0:
                    continue


                metadata = (
                    self.store.metadata[
                        feature_index
                    ]
                )


                hits.append(
                    {
                        "event_id":
                            event_id,

                        "event_order":
                            event_index + 1,

                        "event_text":
                            event_texts[
                                event_index
                            ],

                        "rank":
                            rank,

                        "score":
                            float(
                                score
                            ),

                        "feature_index":
                            feature_index,

                        "video_id":
                            metadata[
                                "video_id"
                            ],

                        "keyframe_id":
                            int(
                                metadata[
                                    "keyframe_id"
                                ]
                            ),

                        "frame_id":
                            int(
                                metadata[
                                    "frame_id"
                                ]
                            ),

                        "pts_time":
                            float(
                                metadata[
                                    "pts_time"
                                ]
                            ),

                        "fps":
                            float(
                                metadata.get(
                                    "fps",
                                    30.0
                                )
                            )
                    }
                )


            results.append(
                {
                    "event_id":
                        event_id,

                    "event_order":
                        event_index + 1,

                    "event_text":
                        event_texts[
                            event_index
                        ],

                    "hits":
                        hits
                }
            )


            if debug:

                print()

                print(
                    f"E{event_index + 1} "
                    f"hits = {len(hits)}"
                )

                for hit in hits[:5]:

                    print(
                        f"    {hit['video_id']} "
                        f"| frame={hit['frame_id']} "
                        f"| time={hit['pts_time']:.2f} "
                        f"| score={hit['score']:.4f}"
                    )


        return results
