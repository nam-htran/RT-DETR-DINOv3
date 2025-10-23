import os
import subprocess
import shutil
import argparse
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.absolute()))
import config

from scripts.convert_coco_to_yolo import run_conversion as run_coco_to_yolo_conversion
from scripts.generate_rtdetr_configs import run_config_generation
from scripts.convert_lightly_checkpoint import run_conversion as run_lightly_checkpoint_conversion
from src.finetune.trainer_yolo import train_yolo_baseline

def run_ddp_command(module_path: str, cwd: Path = None):
    """
    A dedicated helper function to execute DDP commands using torch.distributed.run.
    This is the modern and recommended way to launch distributed training.
    """
    # Use the same Python interpreter that is running this script.
    command = (
        f"{sys.executable} -m torch.distributed.run "
        f"--nproc_per_node={config.NUM_GPUS_PER_NODE} "
        f"{module_path}"
    )
    
    print(f"\n{'='*30}")
    print(f"🚀 Executing DDP command in '{cwd or config.ROOT_DIR}':")
    print(f"   $ {command}")
    print(f"{'='*30}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        # Stream output directly to the console.
        subprocess.run(
            command, shell=True, check=True, text=True,
            cwd=cwd or config.ROOT_DIR, env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: DDP Command failed with exit code {e.returncode} ---", file=sys.stderr)
        if hasattr(e, 'stderr') and e.stderr: 
            print(f"--- STDERR ---\n{e.stderr}", file=sys.stderr)
        raise e

# --- PIPELINE STEPS REFACTORED AS FUNCTIONS ---

def step_prepare_data():
    """STEP 1: Prepare dataset — convert from COCO to YOLO format."""
    print("### STEP 1: Preparing dataset (COCO → YOLO format)... ###")
    run_coco_to_yolo_conversion()

def step_distill():
    """STEP 2: Run knowledge distillation using DDP."""
    print("### STEP 2.1: Running Knowledge Distillation with ConvNeXt Teacher... ###")
    run_ddp_command("-m src.distillation.trainer_convnext")
    
    print("### STEP 2.2: Running Knowledge Distillation with ViT Teacher... ###")
    run_ddp_command("-m src.distillation.trainer_vit")

def step_convert_checkpoint():
    """STEP 3: Convert the ViT checkpoint from Lightly format to RT-DETR format."""
    print("### STEP 3: Converting ViT-distilled checkpoint... ###")
    run_lightly_checkpoint_conversion()

def step_finetune():
    """STEP 4: Run fine-tuning experiments for RT-DETR and YOLO models."""
    
    print("### Generating latest RT-DETR config files from templates... ###")
    run_config_generation()

    RTDETR_SOURCE_PATH = config.RTDETR_SOURCE_DIR
    
    # --- RT-DETR Fine-tuning ---
    print("\n### STEP 4.1: Fine-tuning RT-DETR (ConvNeXt Distilled)... ###")
    convnext_weights_dst = RTDETR_SOURCE_PATH / config.CONVNEXT_BEST_WEIGHTS.name
    print(f"   -> Copying weights from {config.CONVNEXT_BEST_WEIGHTS} to {convnext_weights_dst}")
    shutil.copy(config.CONVNEXT_BEST_WEIGHTS, convnext_weights_dst)
    
    cmd_convnext = (
        f"-m rtdetr.tools.train -c {config.RTDETR_FINETUNE_CONFIG_CONVNEXT.absolute()} --use-amp --seed=0"
    )
    run_ddp_command(cmd_convnext)

    print("\n### STEP 4.2: Fine-tuning RT-DETR (ViT Distilled)... ###")
    vit_weights_dst = RTDETR_SOURCE_PATH / config.VIT_CONVERTED_WEIGHTS.name
    print(f"   -> Copying weights from {config.VIT_CONVERTED_WEIGHTS} to {vit_weights_dst}")
    shutil.copy(config.VIT_CONVERTED_WEIGHTS, vit_weights_dst)
    
    cmd_vit = (
        f"-m rtdetr.tools.train -c {config.RTDETR_FINETUNE_CONFIG_VIT.absolute()} --use-amp --seed=0"
    )
    run_ddp_command(cmd_vit)

    print("\n### STEP 4.3: Fine-tuning RT-DETR (Baseline)... ###")
    cmd_baseline = (
        f"-m rtdetr.tools.train -c {config.RTDETR_FINETUNE_CONFIG_BASELINE.absolute()} --use-amp --seed=0"
    )
    run_ddp_command(cmd_baseline)

    # --- YOLO Fine-tuning ---
    print("\n### STEP 4.4: Fine-tuning YOLO (Baseline)... ###")
    train_yolo_baseline()

def main():
    """Main entry point — orchestrates the full training pipeline."""
    
    print("### Initializing: Installing the project in editable mode... ###")
    subprocess.run(f"{sys.executable} -m pip install -e .", shell=True, check=True)
    
    parser = argparse.ArgumentParser(description="Master training orchestrator for the full ML pipeline.")
    parser.add_argument('--all', action='store_true', help='Run all steps in the pipeline.')
    parser.add_argument('--prepare-data', action='store_true', help='Run only the data preparation step.')
    parser.add_argument('--distill', action='store_true', help='Run only the knowledge distillation step.')
    parser.add_argument('--convert', action='store_true', help='Run only the ViT checkpoint conversion step.')
    parser.add_argument('--finetune', action='store_true', help='Run only the fine-tuning experiments.')
    
    args = parser.parse_args()

    run_all = not any([args.prepare_data, args.distill, args.convert, args.finetune]) or args.all

    if run_all or args.prepare_data:
        step_prepare_data()

    if run_all or args.distill:
        step_distill()

    if run_all or args.convert:
        step_convert_checkpoint()

    if run_all or args.finetune:
        step_finetune()

    print("\nAll selected processes completed successfully.")

if __name__ == "__main__":
    main()