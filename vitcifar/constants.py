"""Central constants for this repository.

Scripts and classes read their defaults from here so that one value has one home.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
PROJECT1_RUNS_DIR = RUNS_DIR / "01_vit_from_scratch"
PROJECT2_RUNS_DIR = RUNS_DIR / "02_attention_maps"
PROJECT2_DEMO_DIR = REPO_ROOT / "projects" / "02_attention_maps" / "demo"
RESULT_SUFFIX = ".json"
CHECKPOINT_SUFFIX = ".pt"

# --- General ------------------------------------------------------------------
DEFAULT_SEED = 0
DEFAULT_NUM_WORKERS = 6
BYTES_PER_MIB = 1024 * 1024

# --- CIFAR-10 -----------------------------------------------------------------
CIFAR10_IMAGE_SIZE = 32
CIFAR10_NUM_CLASSES = 10
CIFAR10_CROP_PADDING = 4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# --- RandAugment --------------------------------------------------------------
RANDAUGMENT_NUM_OPS = 2
RANDAUGMENT_MAGNITUDE = 9

# --- Project 1: ViT from scratch on CIFAR-10 ----------------------------------
P1_EPOCHS = 100
P1_BATCH_SIZE = 512
P1_VIT_DIM = 384
P1_VIT_DEPTH = 7
P1_VIT_HEADS = 6
P1_VIT_MLP_RATIO = 4.0
P1_NAIVE_PATCH_SIZE = 8
P1_SMALL_PATCH_SIZE = 4
P1_VIT_LR = 1e-3
P1_VIT_WEIGHT_DECAY = 0.05
P1_WARMUP_EPOCHS = 10
P1_RESNET_LR = 0.1
P1_RESNET_WEIGHT_DECAY = 5e-4
P1_SGD_MOMENTUM = 0.9
P1_GRAD_CLIP_NORM = 1.0
P1_LABEL_SMOOTHING = 0.1
P1_MIXUP_ALPHA = 0.8
P1_CUTMIX_ALPHA = 1.0
P1_DROP_PATH = 0.1
P1_LONG_SCHEDULE_SCALE = 2.0
P1_SEEDS = (0, 1, 2)
P1_SMOKE_EPOCHS = 2
P1_SMOKE_SUBSET = 2048
P1_FINAL_VIT_STEP = "vit_p4_long"

# Project 1 side study: learning rate with and without warmup
P1_SWEEP_STEP = "vit_p4_aug"
P1_SWEEP_LRS = (5e-4, 1e-3, 2e-3, 4e-3)
P1_SWEEP_EPOCHS = 30
P1_SWEEP_WARMUP_EPOCHS = 5
P1_SWEEP_SMOKE_EPOCHS = 2

# Project 1 side study: accuracy against training-set size
P1_DATA_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
P1_DATA_STEPS = ("resnet18_aug", "vit_p4_droppath")

# --- Project 2: attention maps ------------------------------------------------
P2_NUM_IMAGES = 10
P2_PLOT_IMAGES = 4
P2_ROLLOUT_RESIDUAL = 0.5
P2_DEMO_UPSCALE = 8
P2_STATS_IMAGES = 1000  # test images averaged for the head statistics
P2_STATS_BATCH = 250
QUANT_LEVELS = 255

# --- Plotting -----------------------------------------------------------------
FIG_SIZE = (16, 9)
FIG_DPI = 150
PALETTE = (
    "#E63946", "#1D77B4", "#2A9D8F", "#F4A261", "#8E44AD",
    "#264653", "#E9C46A", "#A0522D", "#06B6D4", "#65A30D",
)
RESNET_COLOR = "#1D77B4"
VIT_COLOR = "#E63946"
