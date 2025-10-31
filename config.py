# ===== config.py (Corrected Version) =====
import os
from pathlib import Path

# --- Environment Configuration ---
os.environ["USE_LIBUV"] = "0"

# --- Core Directories ---
ROOT_DIR = Path(__file__).parent.absolute()
DATA_INPUT_DIR = ROOT_DIR / 'data_input'
PROJECT_OUTPUT_DIR = ROOT_DIR / 'output'

# --- COCO Input Data Source ---
COCO_INPUT_DIR = DATA_INPUT_DIR / 'processed_taco_coco'
COCO_TRAIN_IMAGES = COCO_INPUT_DIR / 'train2017'
COCO_VAL_IMAGES = COCO_INPUT_DIR / 'val2017'
COCO_TRAIN_ANNOTATIONS = COCO_INPUT_DIR / 'annotations/instances_train2017.json'
COCO_VAL_ANNOTATIONS = COCO_INPUT_DIR / 'annotations/instances_val2017.json'

# --- YOLO Data and Outputs ---
YOLO_GROUP_DIR = PROJECT_OUTPUT_DIR / 'YOLO'
YOLO_DATA_DIR = YOLO_GROUP_DIR / 'taco_yolo'
YOLO_TRAIN_IMAGES = YOLO_DATA_DIR / 'images/train'
YOLO_VAL_IMAGES = YOLO_DATA_DIR / 'images/val'
YOLO_TRAIN_LABELS = YOLO_DATA_DIR / 'labels/train'
YOLO_VAL_LABELS = YOLO_DATA_DIR / 'labels/val'
YOLO_CONFIG_FILE = YOLO_DATA_DIR / 'taco.yaml'
YOLO_FINETUNE_OUTPUT_DIR = YOLO_GROUP_DIR / 'yolo_checkpoints'

# --- ConvNeXt Distillation Outputs ---
CONVNEXT_DISTILL_DIR = PROJECT_OUTPUT_DIR / 'DISTILL-CONVNEXT'
CONVNEXT_BEST_WEIGHTS = CONVNEXT_DISTILL_DIR / 'distilled_rtdetr_convnext_teacher_BEST.pth'
CONVNEXT_FINAL_WEIGHTS = CONVNEXT_DISTILL_DIR / 'distilled_rtdetr_convnext_teacher_FINAL.pth'

# --- ViT Distillation Outputs ---
VIT_DISTILL_DIR = PROJECT_OUTPUT_DIR / 'DISTILL-VIT'
VIT_BEST_WEIGHTS = VIT_DISTILL_DIR / 'distilled_rtdetr_vit_teacher_BEST.pth'
VIT_FINAL_WEIGHTS = VIT_DISTILL_DIR / 'distilled_rtdetr_vit_teacher_FINAL.pth'

# --- RT-DETR Source Repo and Generated Configs ---
RTDETR_SOURCE_DIR = ROOT_DIR / 'rtdetr'
RTDETR_PYTORCH_DIR = RTDETR_SOURCE_DIR / 'rtdetrv2_pytorch' # Thêm dòng này để dễ tham chiếu
RTDETR_CONFIG_DIR = RTDETR_PYTORCH_DIR / 'configs/rtdetrv2' # Thêm dòng này để dễ tham chiếu
RTDETR_FINETUNE_CONFIG_CONVNEXT = RTDETR_CONFIG_DIR / 'rtdetrv2_taco_finetune_convnext.yml'
RTDETR_FINETUNE_CONFIG_VIT = RTDETR_CONFIG_DIR / 'rtdetrv2_taco_finetune_vit.yml'
RTDETR_FINETUNE_CONFIG_BASELINE = RTDETR_CONFIG_DIR / 'rtdetrv2_taco_finetune_BASELINE.yml'

# --- Fine-tuning Output Directories ---
FINETUNE_BASELINE_OUTPUT_DIR = PROJECT_OUTPUT_DIR / 'FINETUNE_BASELINE'
FINETUNE_DISTILLED_OUTPUT_DIR = PROJECT_OUTPUT_DIR / 'FINETUNE_DISTILLED'

# --- Source Code and Script Directories ---
SCRIPTS_DIR = ROOT_DIR / 'scripts'
SRC_DIR = ROOT_DIR / 'src'
TEMPLATES_DIR = ROOT_DIR / 'templates' # Mặc dù không dùng nữa nhưng cứ để đây

# --- WandB Project Names ---
WANDB_PROJECT_CONVNEXT_DISTILL = "Distill-RTDETR-ConvNeXt-Teacher"
WANDB_PROJECT_VIT_DISTILL = "Distill-RTDETR-DINOv3-ViT-Teacher"
WANDB_PROJECT_YOLO_FINETUNE = "yolo_runs_taco"

# --- Training Hardware Configuration ---
NUM_GPUS_PER_NODE = 1  # Adjust this to the number of GPUs available```
