# =============================================================================
# config.py
# Master Configuration File — Face Anti-Spoofing Research Project
# Transfer Learning Depth Analysis on Lightweight Models
#
# ALL settings for ALL phases live here.
# No hardcoding is permitted in any phase or utility file.
# Every phase imports this file and reads settings from it.
# =============================================================================

import os

# =============================================================================
# SECTION 1 — PROJECT ROOT
# Set BASE_DIR to the absolute path of your FAS_Project folder.
# All other paths are derived from this single setting.
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# SECTION 2 — DATASET PATHS
# Each dataset must follow this exact structure:
#   DATASET_ROOT/
#       train/real/    train/spoof/
#       val/real/      val/spoof/
#       test/real/     test/spoof/
# =============================================================================

DATASET_ROOT = r"K:\S_dataSet"

DATASETS = {
    "OULU_NPU": os.path.join(DATASET_ROOT, "\S_dataSet\OULU_npu_final_images\OULU_npu_faces\OULU_npu_images1"),
    "Replay_Attack": os.path.join(DATASET_ROOT, "ReplyAttackSplitFaceExtracted"),
    "MSU_MFSD": os.path.join(DATASET_ROOT, "MSU-MFSD_faces"),
    "CelebA_Spoof": os.path.join(DATASET_ROOT, "CelebA_Spoof_Processed"),
    "CASIA_FASD"   :os.path.join(DATASET_ROOT, "CASIA_FASD_processed"),
}

# Subfolder names inside each dataset
SPLIT_NAMES = {
    "train": "train",
    "val": "val",
    "test": "test",
}

# Class folder names (confirmed)
CLASS_FOLDERS = {
    "real": "real",    # label 0
    "spoof": "spoof",  # label 1
}

# Label mapping
LABEL_MAP = {
    "real": 0,
    "spoof": 1,
}

# Processed dataset root (output of Phase 1 face alignment)
PROCESSED_DATASET_ROOT = os.path.join(BASE_DIR, "dataset_processed")

PROCESSED_DATASETS = {
    name: os.path.join(PROCESSED_DATASET_ROOT, name)
    for name in DATASETS.keys()
}

# =============================================================================
# SECTION 3 — MODEL SETTINGS
# Six models covering five distinct architectural philosophies.
# Each entry defines: display name, loader source, timm/torchvision name,
# and the classifier attribute name for replacement.
# =============================================================================

MODELS = {

    "mobilenetv4_conv_small": {
        "display_name"      : "MobileNetV4",
        "source"            : "timm",
        "timm_name"         : "mobilenetv4_conv_small",
        "classifier_attr"   : "classifier",      # timm standard
        "architecture_group": "Depthwise Separable Evolution",
        "year"              : 2024,
    },

    "shufflenet_v2_x2_0": {
        "display_name"      : "ShuffleNetV2",
        "source"            : "torchvision",
        "timm_name"         : None,              # NOT in timm
        "classifier_attr"   : "fc",              # torchvision standard
        "architecture_group": "Channel Splitting",
        "year"              : 2018,
    },

    "ghostnetv2_100": {
        "display_name"      : "GhostNetV2",
        "source"            : "timm",
        "timm_name"         : "ghostnetv2_100",
        "classifier_attr"   : "classifier",
        "architecture_group": "Feature Reuse",
        "year"              : 2022,
    },

    "efficientnet_b0": {
        "display_name"      : "EfficientNet-B0",
        "source"            : "timm",
        "timm_name"         : "efficientnet_b0",
        "classifier_attr"   : "classifier",
        "architecture_group": "NAS Compound Scaling",
        "year"              : 2019,
    },

    "convnext_femto": {
        "display_name"      : "ConvNeXt-Femto",
        "source"            : "timm",
        "timm_name"         : "convnext_femto",
        "classifier_attr"   : "head.fc",         # nested attribute
        "architecture_group": "Modernized Pure CNN",
        "year"              : 2022,
    },

    "mobilevit_xxs": {
        "display_name"      : "MobileViT-XXS",
        "source"            : "timm",
        "timm_name"         : "mobilevit_xxs",
        "classifier_attr"   : "head.fc",         # nested attribute
        "architecture_group": "Hybrid CNN-Transformer",
        "year"              : 2022,
    },
    "resnet50": {
        "display_name"      : "ResNet50",
        "source"            : "torchvision",
        "timm_name"         : None,
        "classifier_attr"   : "fc",
        "architecture_group": "Heavyweight Baseline",
        "year"              : 2016,
    },
}

# Ordered list for consistent iteration across all phases
MODEL_NAMES = list(MODELS.keys())
# Separate list keeps baseline distinct from lightweight models
BASELINE_MODELS = ["resnet50"]

# For running all experiments including baseline
ALL_MODELS = MODEL_NAMES + BASELINE_MODELS

# Number of output classes (binary: real vs spoof)
NUM_CLASSES = 2

# =============================================================================
# SECTION 4 — FINE-TUNING DEPTH LEVELS
#
# L1 — Classifier Adaptation (CA):
#      Freeze entire backbone. Train classifier head only.
#
# L2 — Partial Fine-Tuning (PFT):
#      Freeze first 70% of parameter groups.
#      Train last 30% (last block) + classifier.
#
# L3 — Full Fine-Tuning (FFT):
#      Train all layers using differential learning rates.
#      Backbone uses BACKBONE_LR, classifier uses CLASSIFIER_LR.
# =============================================================================

FT_LEVELS = ["L1", "L2", "L3"]

# Fraction of layers to freeze for L2
L2_FREEZE_RATIO = 0.70

# =============================================================================
# SECTION 5 — TRAINING HYPERPARAMETERS
# =============================================================================

# Input image size for all models
IMAGE_SIZE = 224

# Minimum batch size — MobileNetV4 requires >= 2 for BatchNorm
BATCH_SIZE = 16

# Number of training epochs
NUM_EPOCHS = 30

# Early stopping: stop if val metric does not improve for this many epochs
EARLY_STOPPING_PATIENCE = 10

# Optimizer settings
OPTIMIZER = "adam"          # options: "adam", "sgd", "adamw"
WEIGHT_DECAY = 1e-4

# Learning rates
# L1 and L2: single learning rate (backbone frozen or partially frozen)
LEARNING_RATE = 1e-3

# L3 differential learning rates
BACKBONE_LR   = 5e-5        # lower LR for pre-trained backbone layers
CLASSIFIER_LR = 1e-3        # higher LR for classifier head

# Loss function
LOSS_FUNCTION = "cross_entropy"   # standard for binary classification

# Number of DataLoader worker processes
NUM_WORKERS = 1

# Pin memory for faster GPU transfer
PIN_MEMORY = True

# =============================================================================
# SECTION 6 — DATA AUGMENTATION
# Applied to TRAINING set only.
# Validation and test sets use resize + normalize only.
# =============================================================================

# ImageNet normalization statistics (used for all models)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# Training augmentation settings
AUGMENTATION = {
    "horizontal_flip_prob" : 0.5,
    "rotation_degrees"     : 10,
    "brightness_jitter"    : 0.2,
    "contrast_jitter"      : 0.2,
    "saturation_jitter"    : 0.1,
    "random_crop_scale"    : (0.8, 1.0),   # RandomResizedCrop scale range
}

# =============================================================================
# SECTION 7 — PHASE 1: FACE DETECTION & ALIGNMENT SETTINGS
# =============================================================================

# Face detector choice: "mtcnn" or "retinaface"
FACE_DETECTOR = "mtcnn"

# MTCNN settings
MTCNN_IMAGE_SIZE       = 224
MTCNN_MARGIN           = 30       # pixels of margin around detected face
MTCNN_MIN_FACE_SIZE    = 40       # minimum face size in pixels
MTCNN_THRESHOLDS       = [0.6, 0.7, 0.7]   # P-Net, R-Net, O-Net thresholds
MTCNN_KEEP_ALL         = False    # keep only the largest/most confident face

# Alignment output size
ALIGNED_IMAGE_SIZE = 224

# Bounding box expansion factor (20% as per Martínez-Díaz et al. CVPRW 2023)
BBOX_EXPANSION_FACTOR = 0.20

# If face not detected in image, save to failure log and skip image
SKIP_ON_DETECTION_FAILURE = True

# =============================================================================
# SECTION 8 — PROTOCOL B: CROSS-DATASET CONFIGURATIONS
#
# Following Almeida et al. (2025) unbiased benchmark:
#   - Validation set = val split of the TEST dataset
#   - Test set is evaluated exactly ONCE after training completes
#   - Train set = train + val splits of the TRAINING datasets
# =============================================================================

PROTOCOL_B_CONFIGS = {

    # ── Original 3-dataset configs (keep for backward compatibility) ──

    "RpM_to_O": {
        "display_name"  : "R+M → O",
        "train_datasets": [
            {"dataset": "Replay_Attack", "splits": ["train", "val"]},
            {"dataset": "MSU_MFSD",      "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "OULU_NPU",     "splits": ["val"]},
        "test_dataset"  : {"dataset": "OULU_NPU",     "splits": ["test"]},
    },

    "OpM_to_R": {
        "display_name"  : "O+M → R",
        "train_datasets": [
            {"dataset": "OULU_NPU", "splits": ["train", "val"]},
            {"dataset": "MSU_MFSD", "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "Replay_Attack", "splits": ["val"]},
        "test_dataset"  : {"dataset": "Replay_Attack", "splits": ["test"]},
    },

    "OpR_to_M": {
        "display_name"  : "O+R → M",
        "train_datasets": [
            {"dataset": "OULU_NPU",      "splits": ["train", "val"]},
            {"dataset": "Replay_Attack", "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "MSU_MFSD", "splits": ["val"]},
        "test_dataset"  : {"dataset": "MSU_MFSD", "splits": ["test"]},
    },

    # ── NEW: Full O-C-I-M four-dataset protocol ──

    "CpIpM_to_O": {
        "display_name"  : "C+I+M → O",
        "train_datasets": [
            {"dataset": "CASIA_FASD",    "splits": ["train", "val"]},
            {"dataset": "Replay_Attack", "splits": ["train", "val"]},
            {"dataset": "MSU_MFSD",      "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "OULU_NPU", "splits": ["val"]},
        "test_dataset"  : {"dataset": "OULU_NPU", "splits": ["test"]},
    },

    "OpIpM_to_C": {
        "display_name"  : "O+I+M → C",
        "train_datasets": [
            {"dataset": "OULU_NPU",      "splits": ["train", "val"]},
            {"dataset": "Replay_Attack", "splits": ["train", "val"]},
            {"dataset": "MSU_MFSD",      "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "CASIA_FASD", "splits": ["val"]},
        "test_dataset"  : {"dataset": "CASIA_FASD", "splits": ["test"]},
    },

    "OpCpM_to_I": {
        "display_name"  : "O+C+M → I",
        "train_datasets": [
            {"dataset": "OULU_NPU",   "splits": ["train", "val"]},
            {"dataset": "CASIA_FASD", "splits": ["train", "val"]},
            {"dataset": "MSU_MFSD",   "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "Replay_Attack", "splits": ["val"]},
        "test_dataset"  : {"dataset": "Replay_Attack", "splits": ["test"]},
    },

    "OpCpI_to_M": {
        "display_name"  : "O+C+I → M",
        "train_datasets": [
            {"dataset": "OULU_NPU",      "splits": ["train", "val"]},
            {"dataset": "CASIA_FASD",    "splits": ["train", "val"]},
            {"dataset": "Replay_Attack", "splits": ["train", "val"]},
        ],
        "val_dataset"   : {"dataset": "MSU_MFSD", "splits": ["val"]},
        "test_dataset"  : {"dataset": "MSU_MFSD", "splits": ["test"]},
    },
}

PROTOCOL_B_CONFIG_NAMES = list(PROTOCOL_B_CONFIGS.keys())

# =============================================================================
# SECTION 9 — METRICS SETTINGS
# =============================================================================

# Threshold range for EER computation (linspace from 0 to 1)
EER_THRESHOLD_STEPS = 1000

# Primary metric used for:
#   - Best checkpoint selection in Protocol A
PROTOCOL_A_PRIMARY_METRIC = "EER"        # lower is better

#   - Best checkpoint selection in Protocol B
PROTOCOL_B_PRIMARY_METRIC = "HTER"       # lower is better

# All metrics to compute and save in every experiment
METRICS_TO_COMPUTE = [
    "EER",
    "HTER",
    "APCER",
    "BPCER",
    "ACER",
    "AUC",
    "Accuracy",
    "Inference_ms",
]

# =============================================================================
# SECTION 10 — PHASE 5: ABLATION STUDY SETTINGS
# =============================================================================

# Best model will be determined automatically from Phase 3 results
# (lowest average EER across all datasets)
# Can be manually overridden here if needed
ABLATION_MODEL_OVERRIDE = None   # e.g. "efficientnet_b0" or None for auto

# Ablation 1: Learning rate sensitivity
ABLATION_LEARNING_RATES = [1e-3, 1e-4]

# Ablation 2: Differential vs uniform LR at L3
# Uniform LR uses LEARNING_RATE for all layers
# Differential LR uses BACKBONE_LR + CLASSIFIER_LR
ABLATION_LR_STRATEGIES = ["uniform", "differential"]

# =============================================================================
# SECTION 11 — PHASE 6: VISUALIZATION SETTINGS
# =============================================================================

# Number of sample images per class for Grad-CAM
GRADCAM_SAMPLES_PER_CLASS = 5

# Number of samples for t-SNE embedding extraction
TSNE_SAMPLES_PER_CLASS = 200

# t-SNE hyperparameters
TSNE_PERPLEXITY   = 30
TSNE_N_ITER       = 1000
TSNE_RANDOM_STATE = 42

# Grad-CAM target layer name per model
# These are the last convolutional layers before global average pooling
GRADCAM_TARGET_LAYERS = {
    "mobilenetv4_conv_small": "blocks",
    "shufflenet_v2_x2_0"    : "conv5",
    "ghostnetv2_100"        : "blocks",
    "efficientnet_b0"       : "blocks",
    "convnext_femto"        : "stages",
    "mobilevit_xxs"         : "stages",
}

# =============================================================================
# SECTION 12 — OUTPUT PATHS
# All results, checkpoints, and logs are organized under BASE_DIR.
# =============================================================================

# Results directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RESULTS_PHASE1  = os.path.join(RESULTS_DIR, "phase1_data_prep")
RESULTS_PHASE2  = os.path.join(RESULTS_DIR, "phase2_model_setup")
RESULTS_PHASE3  = os.path.join(RESULTS_DIR, "phase3_protocol_a")
RESULTS_PHASE4  = os.path.join(RESULTS_DIR, "phase4_protocol_b")
RESULTS_PHASE5  = os.path.join(RESULTS_DIR, "phase5_ablation")
RESULTS_PHASE6  = os.path.join(RESULTS_DIR, "phase6_visualization")
RESULTS_PHASE7  = os.path.join(RESULTS_DIR, "phase7_summary")

GRADCAM_DIR = os.path.join(RESULTS_PHASE6, "gradcam")
TSNE_DIR    = os.path.join(RESULTS_PHASE6, "tsne")

# Checkpoint directory
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# =============================================================================
# SECTION 13 — CSV COLUMN DEFINITIONS
# Defines exact column names for every results CSV.
# Phase 7 relies on these names for aggregation.
# =============================================================================

# Phase 3 — Protocol A CSV columns
CSV_COLUMNS_PROTOCOL_A = [
    "experiment_id",    # unique key: model__ftlevel__dataset
    "model",
    "ft_level",
    "dataset",
    "EER",
    "HTER",
    "APCER",
    "BPCER",
    "ACER",
    "AUC",
    "Accuracy",
    "Inference_ms",
    "best_epoch",
    "total_epochs",
    "timestamp",
]

# Phase 4 — Protocol B CSV columns
CSV_COLUMNS_PROTOCOL_B = [
    "experiment_id",    # unique key: model__ftlevel__config
    "model",
    "ft_level",
    "config_name",
    "train_datasets",
    "val_dataset",
    "test_dataset",
    "val_EER",
    "test_HTER",
    "test_APCER",
    "test_BPCER",
    "test_ACER",
    "test_AUC",
    "test_Accuracy",
    "Inference_ms",
    "best_epoch",
    "total_epochs",
    "timestamp",
]

# Phase 5 — Ablation CSV columns
CSV_COLUMNS_ABLATION_LR = [
    "experiment_id",
    "model",
    "ft_level",
    "dataset",
    "learning_rate",
    "lr_strategy",
    "EER",
    "HTER",
    "APCER",
    "BPCER",
    "ACER",
    "AUC",
    "Accuracy",
    "best_epoch",
    "timestamp",
]

# Phase 2 — Model layer report CSV columns
CSV_COLUMNS_MODEL_REPORT = [
    "model",
    "ft_level",
    "total_params_M",
    "trainable_params_M",
    "frozen_params_M",
    "trainable_percent",
    "GFLOPs",
    "inference_ms_cpu",
    "status",
]

# =============================================================================
# SECTION 14 — DEVICE SETTINGS
# =============================================================================

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For reproducibility across all experiments
RANDOM_SEED = 42

# =============================================================================
# SECTION 15 — LOGGING SETTINGS
# =============================================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Whether to also print logs to console
LOG_TO_CONSOLE = True

# =============================================================================
# SECTION 16 — DIRECTORY CREATION
# Automatically creates all output directories when config is imported.
# Safe to call multiple times (exist_ok=True).
# =============================================================================

def create_output_dirs():
    """Create all output directories defined in config."""
    dirs = [
        RESULTS_DIR,
        RESULTS_PHASE1,
        RESULTS_PHASE2,
        RESULTS_PHASE3,
        RESULTS_PHASE4,
        RESULTS_PHASE5,
        RESULTS_PHASE6,
        RESULTS_PHASE7,
        GRADCAM_DIR,
        TSNE_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        PROCESSED_DATASET_ROOT,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# Auto-create directories on import
create_output_dirs()


# =============================================================================
# SECTION 17 — CONFIG VALIDATION
# Called by main.py at startup to catch configuration errors early.
# =============================================================================

def validate_config():
    """
    Validate all critical config settings.
    Raises ValueError with a clear message if anything is wrong.
    Returns True if all checks pass.
    """
    errors = []

    # Check dataset paths exist
    for name, path in DATASETS.items():
        if not os.path.isdir(path):
            errors.append(
                f"Dataset folder not found: {name} -> {path}\n"
                f"  Please update DATASET_ROOT in config.py"
            )

    # Check batch size constraint for MobileNetV4 BatchNorm
    if BATCH_SIZE < 2:
        errors.append(
            f"BATCH_SIZE={BATCH_SIZE} is too small. "
            f"MobileNetV4 requires BATCH_SIZE >= 2 due to BatchNorm."
        )

    # Check learning rate sanity
    if LEARNING_RATE <= 0:
        errors.append(f"LEARNING_RATE must be positive. Got {LEARNING_RATE}")

    if BACKBONE_LR >= CLASSIFIER_LR:
        errors.append(
            f"For L3 differential LR, BACKBONE_LR ({BACKBONE_LR}) should be "
            f"less than CLASSIFIER_LR ({CLASSIFIER_LR})."
        )

    # Check L2 freeze ratio
    if not (0.0 < L2_FREEZE_RATIO < 1.0):
        errors.append(
            f"L2_FREEZE_RATIO must be between 0 and 1. Got {L2_FREEZE_RATIO}"
        )

    # Check model names are valid
    valid_model_names = list(MODELS.keys())
    for name in MODEL_NAMES:
        if name not in valid_model_names:
            errors.append(f"MODEL_NAMES contains unknown model: {name}")

    # Check FT levels
    valid_ft = ["L1", "L2", "L3"]
    for lvl in FT_LEVELS:
        if lvl not in valid_ft:
            errors.append(f"FT_LEVELS contains unknown level: {lvl}")

    if errors:
        msg = "\n\nCONFIG VALIDATION FAILED:\n" + "\n".join(
            f"  [{i+1}] {e}" for i, e in enumerate(errors)
        )
        raise ValueError(msg)

    return True


# =============================================================================
# SECTION 18 — HELPER: EXPERIMENT ID GENERATOR
# Used by all phases to create a unique, consistent experiment identifier.
# This ID is used for:
#   - CSV row identification
#   - Checkpoint file naming
#   - Skip/resume logic
# =============================================================================

def get_experiment_id(model: str, ft_level: str, identifier: str) -> str:
    """
    Generate a unique experiment ID.

    Args:
        model      : model key from MODELS dict
        ft_level   : one of 'L1', 'L2', 'L3'
        identifier : dataset name (Phase 3) or config name (Phase 4/5)

    Returns:
        str: e.g. 'efficientnet_b0__L2__OULU_NPU'
    """
    return f"{model}__{ft_level}__{identifier}"


def get_checkpoint_path(model: str, ft_level: str, identifier: str) -> str:
    """
    Return the full path for saving/loading a model checkpoint.

    Args:
        model      : model key from MODELS dict
        ft_level   : one of 'L1', 'L2', 'L3'
        identifier : dataset name or config name

    Returns:
        str: full path to .pth file
    """
    exp_id = get_experiment_id(model, ft_level, identifier)
    folder = os.path.join(CHECKPOINTS_DIR, exp_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "best_model.pth")


# =============================================================================
# QUICK SELF-TEST
# When config.py is run directly, print a summary of all settings.
# Usage: python config.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("FAS PROJECT — CONFIG SUMMARY")
    print("=" * 65)

    print(f"\n[DEVICE]")
    print(f"  Running on : {DEVICE}")
    print(f"  Random seed: {RANDOM_SEED}")

    print(f"\n[DATASETS]")
    for name, path in DATASETS.items():
        exists = "EXISTS" if os.path.isdir(path) else "NOT FOUND"
        print(f"  {name:<20} {exists:<12} {path}")

    print(f"\n[MODELS — {len(MODELS)} total]")
    for key, cfg in MODELS.items():
        print(f"  {key:<30} source={cfg['source']:<14} "
              f"group={cfg['architecture_group']}")

    print(f"\n[FINE-TUNING LEVELS]")
    for lvl in FT_LEVELS:
        print(f"  {lvl}")

    print(f"\n[TRAINING]")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Early stopping  : {EARLY_STOPPING_PATIENCE} epochs patience")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Backbone LR(L3) : {BACKBONE_LR}")
    print(f"  Classifier LR   : {CLASSIFIER_LR}")
    print(f"  Optimizer       : {OPTIMIZER}")

    print(f"\n[PROTOCOL B CONFIGS]")
    for key, cfg in PROTOCOL_B_CONFIGS.items():
        print(f"  {cfg['display_name']}")

    print(f"\n[OUTPUT DIRECTORIES]")
    print(f"  Results    : {RESULTS_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  Logs       : {LOGS_DIR}")
    print(f"  Processed  : {PROCESSED_DATASET_ROOT}")

    print(f"\n[VALIDATION]")
    try:
        validate_config()
        print("  All checks PASSED")
    except ValueError as e:
        print(str(e))

    print("\n" + "=" * 65)
