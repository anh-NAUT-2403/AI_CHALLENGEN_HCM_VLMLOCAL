# config.py


# ============================================================
# DATA PATHS
# ============================================================

FAISS_INDEX_PATH = (
    "/content/drive/MyDrive/"
    "AICHALLENGENHCM2026/"
    "retrieval_index/"
    "aic2026_clip.index"
)


METADATA_PATH = (
    "/content/drive/MyDrive/"
    "AICHALLENGENHCM2026/"
    "retrieval_index/"
    "aic2026_metadata.pkl"
)


OBJECT_ZIP_PATH = (
    "/content/drive/MyDrive/"
    "AICHALLENGENHCM2026/"
    "AI CHALLENGEN HCM2026/"
    "objects-aic25-b1.zip"
)


VIDEOS_ROOT = (
    "/content/drive/MyDrive/"
    "AICHALLENGENHCM2026/"
    "AI CHALLENGEN HCM2026"
)


# ============================================================
# LOCAL COLAB CACHE
# ============================================================

VIDEO_CACHE_ROOT = (
    "/content/video_cache"
)


SEGMENT_ROOT = (
    "/content/video_cache/segments"
)


# ============================================================
# CLIP
# ============================================================

CLIP_MODEL_NAME = (
    "ViT-B/32"
)


# None =
# GPU nếu có, nếu không CPU.
CLIP_DEVICE = None


# ============================================================
# RETRIEVAL
# ============================================================

# Top-K mỗi expanded query.
RETRIEVAL_TOP_K = 50


# Số video Retrieval trả ra.
RETRIEVAL_TOP_N = 20


# ============================================================
# TEMPORAL CLUSTERING
# ============================================================

TEMPORAL_MAX_GAP_SECONDS = 3


TEMPORAL_MAX_CLUSTER_DURATION = 10.0


# ============================================================
# MULTI-QUERY FUSION
# ============================================================

# semantic_score =
#
# 0.6 * best query
# +
# 0.3 * second query
# +
# 0.1 * coverage

FUSION_BEST_WEIGHT = 0.60

FUSION_SECOND_WEIGHT = 0.30

FUSION_COVERAGE_WEIGHT = 0.10


# ============================================================
# OBJECT RETRIEVAL
# ============================================================

USE_OBJECT = True


OBJECT_DETECTION_THRESHOLD = 0.30


OBJECT_SEMANTIC_THRESHOLD = 0.8


# Object chỉ boost semantic retrieval.
#
# final =
# semantic *
# (
#     1
#     +
#     weight * object_boost
# )

OBJECT_BOOST_WEIGHT = 0.25


# ============================================================
# VIDEO SEGMENT
# ============================================================

# Padding quanh temporal region.

SEGMENT_PADDING_BEFORE = 6.0

SEGMENT_PADDING_AFTER = 6.0


# Segment không quá ngắn.

SEGMENT_MIN_DURATION = 6.0


# Không đưa clip quá dài vào VLM.

SEGMENT_MAX_DURATION = 100.0


# ============================================================
# VLM
# ============================================================

USE_VLM = True


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

# Baseline Colab:
VLM_MODEL_NAME = (
    "Qwen/Qwen3-VL-8B-Instruct"
)

# Nếu GPU mạnh hơn có thể thử:
#
# VLM_MODEL_NAME = (
#     "Qwen/Qwen3-VL-8B-Instruct"
# )


VLM_DEVICE_MAP = "auto"

VLM_DTYPE = "auto"


# ------------------------------------------------------------
# VIDEO SAMPLING
# ------------------------------------------------------------

# KIS thường không cần quá nhiều frame.
VLM_FPS_KIS = 2.0


# QA baseline.
VLM_FPS_QA = 1.0


# TRAKE cần temporal detail nhiều hơn.
VLM_FPS_TRAKE = 2.0


# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------

VLM_MAX_NEW_TOKENS = 512


# ------------------------------------------------------------
# HOW MANY RETRIEVAL RESULTS VLM CHECKS
# ------------------------------------------------------------

# Retrieval Top-10
# nhưng chỉ load video + VLM Top-5.

VLM_VERIFY_TOP_N = 5


# ============================================================
# VLM RERANK
# ============================================================

# Match score >= threshold:
# candidate bắt đầu được coi là match.
#
# Nên tune:
# 0.40
# 0.50
# 0.60

VLM_MATCH_THRESHOLD = 0.50


# Maximum boost khi VLM match cực mạnh.
#
# score = 1.0
# → factor tối đa = 1 + 0.50 = 1.50

VLM_POSITIVE_BOOST = 0.60


# Maximum penalty khi VLM cho match_score = 0.
#
# penalty 0.80:
# factor lý thuyết = 0.20

VLM_REJECT_PENALTY = 0.80


# Safeguard:
# không để Retrieval candidate bị nhân factor nhỏ hơn 0.10.

VLM_MINIMUM_FACTOR = 0.10

# ============================================================
# DEBUG
# ============================================================

DEBUG_RETRIEVAL = True

DEBUG_VIDEO_PIPELINE = True
# ============================================================
# QUERY PACKAGE
# ============================================================

# Mỗi đợt BTC đưa file ZIP mới thì chỉ cần sửa path này.
#
# Code cũng hỗ trợ nếu đây là một folder chứa *.txt.

QUERY_PACKAGE_PATH = (
    "/content/drive/MyDrive/AICHALLENGENHCM2026/SOTUYEN1-bo-de-thi.zip"
)


# ============================================================
# SUBMISSION
# ============================================================

# Folder tạm tạo CSV.
SUBMISSION_WORK_ROOT = (
    "/content/aic_submission"
)


# File ZIP cuối cùng để nộp.
SUBMISSION_ZIP_PATH = (
    "/content/submission.zip"
)


# Nếu True:
# chỉ cần 1 query không tạo được output hợp lệ
# thì KHÔNG tạo submission.zip.
#
# Tôi rất khuyên giữ True vì BTC chỉ cho nộp 3 lần.
SUBMISSION_STRICT = True


# ============================================================
# FRAME RESOLUTION
# ============================================================

# "nearest_keyframe":
#   dùng timestamp VLM tìm được,
#   rồi tìm keyframe gần nhất trong metadata.
#
# "candidate":
#   luôn dùng candidate_frame từ Retrieval.
#
# Tôi khuyên nearest_keyframe.
SUBMISSION_FRAME_MODE = (
    "nearest_keyframe"
)


# ============================================================
# NUMBER OF SUBMISSION ROWS
# ============================================================

# BTC cho tối đa 100 dòng.

SUBMISSION_KIS_MAX_LINES = 100

SUBMISSION_QA_MAX_LINES = 100

SUBMISSION_TRAKE_MAX_LINES = 100


# ============================================================
# SUBMISSION VALIDATION
# ============================================================

# Với QA chỉ output candidate mà VLM thực sự xác nhận.
SUBMISSION_QA_REQUIRE_VERIFIED = True


# TRAKE càng cần verification.
SUBMISSION_TRAKE_REQUIRE_VERIFIED = True
# ============================================================
# TRAKE RETRIEVAL
# ============================================================

# Retrieval riêng cho mỗi event.
TRAKE_EVENT_TOP_K = 100


# Số video alignment giữ lại.
TRAKE_ALIGNMENT_TOP_N = 70


# Beam search.
TRAKE_ALIGNMENT_BEAM_SIZE = 20


# Mỗi event / video chỉ giữ các hit tốt nhất.
TRAKE_HITS_PER_EVENT_PER_VIDEO = 20


# Khoảng cách tối đa giữa hai event liên tiếp.
TRAKE_MAX_EVENT_GAP_SECONDS = 10.0


# Tổng span E1 -> EN.
TRAKE_MAX_TOTAL_SPAN_SECONDS = 20.0


# Penalize sequence quá dài nhẹ.
TRAKE_GAP_PENALTY_WEIGHT = 0.03


# ============================================================
# TRAKE SUBMISSION
# ============================================================

# Nếu VLM không verify được nhưng event retrieval
# đã tạo sequence hợp lệ thì vẫn cho phép nộp.
SUBMISSION_TRAKE_ALLOW_RETRIEVAL_FALLBACK = True