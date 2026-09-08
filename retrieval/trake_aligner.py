from collections import defaultdict


class TrakeAligner:

    def __init__(
        self,
        top_n=100,
        beam_size=30,
        hits_per_event_per_video=20,
        max_event_gap_seconds=180.0,
        max_total_span_seconds=120.0,
        gap_penalty_weight=0.03
    ):

        self.top_n = int(
            top_n
        )

        self.beam_size = int(
            beam_size
        )

        self.hits_per_event_per_video = int(
            hits_per_event_per_video
        )

        self.max_event_gap_seconds = (
            max_event_gap_seconds
        )

        self.max_total_span_seconds = (
            max_total_span_seconds
        )

        self.gap_penalty_weight = float(
            gap_penalty_weight
        )


    # ========================================================
    # GROUP BY VIDEO
    # ========================================================

    def _group_by_video(
        self,
        event_results
    ):

        grouped = defaultdict(
            lambda: defaultdict(list)
        )


        for event_result in event_results:

            event_order = int(
                event_result[
                    "event_order"
                ]
            )


            for hit in event_result[
                "hits"
            ]:

                video_id = hit[
                    "video_id"
                ]

                grouped[
                    video_id
                ][
                    event_order
                ].append(
                    hit
                )


        # Sort theo time
        for video_id in grouped:

            for event_order in grouped[
                video_id
            ]:

                grouped[
                    video_id
                ][
                    event_order
                ].sort(

                    key=lambda x:
                        x[
                            "pts_time"
                        ]
                )


        return grouped


    # ========================================================
    # ALIGN ONE VIDEO
    # ========================================================

    def _align_video(
        self,
        video_id,
        event_map,
        event_count
    ):

        # Video phải có hit cho đủ events.
        for event_order in range(
            1,
            event_count + 1
        ):

            if not event_map.get(
                event_order
            ):

                return []


        # ====================================================
        # FIRST EVENT
        # ====================================================

        first_hits = sorted(

            event_map[1],

            key=lambda x:
                x[
                    "score"
                ],

            reverse=True

        )[
            :self.hits_per_event_per_video
        ]


        beams = [

            {
                "sequence":
                    [
                        hit
                    ],

                "score_sum":
                    float(
                        hit[
                            "score"
                        ]
                    )
            }

            for hit
            in first_hits
        ]


        # ====================================================
        # NEXT EVENTS
        # ====================================================

        for event_order in range(
            2,
            event_count + 1
        ):

            event_hits = sorted(

                event_map[
                    event_order
                ],

                key=lambda x:
                    x[
                        "score"
                    ],

                reverse=True

            )[
                :self.hits_per_event_per_video
            ]


            new_beams = []


            for beam in beams:

                previous = (
                    beam[
                        "sequence"
                    ][-1]
                )

                previous_time = float(
                    previous[
                        "pts_time"
                    ]
                )


                for hit in event_hits:

                    current_time = float(
                        hit[
                            "pts_time"
                        ]
                    )


                    # ------------------------------
                    # MUST BE AFTER PREVIOUS EVENT
                    # ------------------------------

                    if (
                        current_time
                        <=
                        previous_time
                    ):

                        continue


                    # ------------------------------
                    # MAX GAP
                    # ------------------------------

                    gap = (
                        current_time
                        -
                        previous_time
                    )


                    if (
                        self.max_event_gap_seconds
                        is not None
                        and
                        gap
                        >
                        self.max_event_gap_seconds
                    ):

                        continue


                    sequence = (
                        beam[
                            "sequence"
                        ]
                        +
                        [
                            hit
                        ]
                    )


                    # ------------------------------
                    # TOTAL SPAN
                    # ------------------------------

                    total_span = (

                        float(
                            sequence[-1][
                                "pts_time"
                            ]
                        )

                        -

                        float(
                            sequence[0][
                                "pts_time"
                            ]
                        )
                    )


                    if (
                        self.max_total_span_seconds
                        is not None
                        and
                        total_span
                        >
                        self.max_total_span_seconds
                    ):

                        continue


                    new_beams.append(
                        {
                            "sequence":
                                sequence,

                            "score_sum":
                                (
                                    beam[
                                        "score_sum"
                                    ]
                                    +
                                    float(
                                        hit[
                                            "score"
                                        ]
                                    )
                                )
                        }
                    )


            if not new_beams:

                return []


            # =================================================
            # KEEP BEST BEAMS
            # =================================================

            new_beams.sort(

                key=lambda x:
                    x[
                        "score_sum"
                    ],

                reverse=True
            )


            beams = new_beams[
                :self.beam_size
            ]


        # ====================================================
        # FINAL SCORE
        # ====================================================

        outputs = []


        for beam in beams:

            sequence = beam[
                "sequence"
            ]


            first_time = float(
                sequence[0][
                    "pts_time"
                ]
            )

            last_time = float(
                sequence[-1][
                    "pts_time"
                ]
            )

            total_span = (
                last_time
                -
                first_time
            )


            semantic_score = (

                beam[
                    "score_sum"
                ]

                /
                event_count
            )


            # Compact sequence được boost nhẹ.
            if (
                self.max_total_span_seconds
                and
                self.max_total_span_seconds > 0
            ):

                normalized_span = min(

                    total_span
                    /
                    self.max_total_span_seconds,

                    1.0
                )

            else:

                normalized_span = 0.0


            alignment_score = (

                semantic_score

                -

                self.gap_penalty_weight
                *
                normalized_span
            )


            aligned_events = []


            for event_order, hit in enumerate(
                sequence,
                start=1
            ):

                aligned_events.append(
                    {
                        "event_id":
                            hit[
                                "event_id"
                            ],

                        "event_order":
                            event_order,

                        "frame_id":
                            int(
                                hit[
                                    "frame_id"
                                ]
                            ),

                        "keyframe_id":
                            int(
                                hit[
                                    "keyframe_id"
                                ]
                            ),

                        "time":
                            float(
                                hit[
                                    "pts_time"
                                ]
                            ),

                        "score":
                            float(
                                hit[
                                    "score"
                                ]
                            )
                    }
                )


            outputs.append(
                {
                    "video_id":
                        video_id,

                    "score":
                        float(
                            alignment_score
                        ),

                    "retrieval_score":
                        float(
                            alignment_score
                        ),

                    "semantic_score":
                        float(
                            semantic_score
                        ),

                    "candidate_frame":
                        int(
                            sequence[0][
                                "frame_id"
                            ]
                        ),

                    "candidate_time":
                        float(
                            (
                                first_time
                                +
                                last_time
                            )
                            /
                            2.0
                        ),

                    "temporal_window":
                        {
                            "start_time":
                                first_time,

                            "end_time":
                                last_time
                        },

                    "aligned_events":
                        aligned_events,

                    "event_count":
                        event_count,

                    "alignment_span":
                        float(
                            total_span
                        )
                }
            )


        return outputs


    # ========================================================
    # ALIGN ALL VIDEOS
    # ========================================================

    def align(
        self,
        event_results,
        debug=False
    ):

        if not event_results:

            return []


        event_count = len(
            event_results
        )


        grouped = self._group_by_video(
            event_results
        )


        candidates = []


        for video_id, event_map in (
            grouped.items()
        ):

            video_alignments = (
                self._align_video(
                    video_id=video_id,
                    event_map=event_map,
                    event_count=event_count
                )
            )

            candidates.extend(
                video_alignments
            )


        # ====================================================
        # GLOBAL RANK
        # ====================================================

        candidates.sort(

            key=lambda x:
                x[
                    "score"
                ],

            reverse=True
        )


        # Một candidate tốt nhất / video trước.
        best_by_video = {}


        for candidate in candidates:

            video_id = candidate[
                "video_id"
            ]

            if video_id not in best_by_video:

                best_by_video[
                    video_id
                ] = candidate


        final = list(
            best_by_video.values()
        )


        final.sort(

            key=lambda x:
                x[
                    "score"
                ],

            reverse=True
        )


        final = final[
            :self.top_n
        ]


        for rank, candidate in enumerate(
            final,
            start=1
        ):

            candidate[
                "rank"
            ] = rank


        if debug:

            print(
                "\n=========================================="
            )

            print(
                "TRAKE ALIGNMENT"
            )

            print(
                "=========================================="
            )

            print(
                "Candidate videos:",
                len(
                    final
                )
            )


            for candidate in final[:10]:

                print()

                print(
                    f"#{candidate['rank']} "
                    f"{candidate['video_id']} "
                    f"| score="
                    f"{candidate['score']:.4f} "
                    f"| span="
                    f"{candidate['alignment_span']:.1f}s"
                )

                for event in (
                    candidate[
                        "aligned_events"
                    ]
                ):

                    print(
                        f"    E{event['event_order']} "
                        f"frame={event['frame_id']} "
                        f"time={event['time']:.2f} "
                        f"score={event['score']:.4f}"
                    )


        return final
