# ===== train.py (Final Workaround Version) =====
import os
import subprocess
import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing 'config'
sys.path.append(str(Path(__file__).parent.absolute()))
import config

# Import pipeline step functions
from scripts.convert_coco_to_yolo import run_conversion as run_coco_to_yolo_conversion
from scripts.generate_rtdetr_configs import run_config_generation
from src.finetune.trainer_yolo import train_yolo_baseline

def run_python_script(script_path: str, cwd: Path):
    """
    Helper function to run a standard Python script.
    """
    command = f"{sys.executable} {script_path}"
    print(f"\n{'='*30}")
    print(f"🚀 Executing command in '{cwd}':")
    print(f"   $ {command}")
    print(f"{'='*30}")
    
    try:
        subprocess.run(command, shell=True, check=True, text=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Command failed with exit code {e.returncode} ---", file=sys.stderr)
        raise e

def run_manual_ddp_script(script_path: str, cwd: Path):
    """
    WORKAROUND for Windows DDP.
    This function manually sets environment variables and launches the training
    script directly, bypassing the problematic torchrun launcher.
    This is only intended for single-GPU or single-node multi-GPU setups.
    """
    # Create a copy of the current environment
    env = os.environ.copy()

    # --- Manually set DDP environment variables ---
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = '29500'  # A free port
    env['RANK'] = '0'
    env['WORLD_SIZE'] = str(config.NUM_GPUS_PER_NODE)
    env['LOCAL_RANK'] = '0' # For single-process launch, this is always 0.
                           # The script itself will handle device placement.
    
    # --- CRITICAL FIX for the original error ---
    env["USE_LIBUV"] = "0"
    
    # Add project root to PYTHONPATH
    env["PYTHONPATH"] = str(config.ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    command = f"{sys.executable} {script_path}"
    
    print(f"\n{'='*30}")
    print(f"🚀 Executing MANUAL DDP command in '{cwd}':")
    print(f"   (Setting DDP ENV VARS and USE_LIBUV=0)")
    print(f"   $ {command}")
    print(f"{'='*30}")

    try:
        # Run the command with the custom environment
        subprocess.run(command, shell=True, check=True, text=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Manual DDP Command failed with exit code {e.returncode} ---", file=sys.stderr)
        raise e

def step_prepare_data():
    """STEP 1: Prepare dataset — convert from COCO to YOLO format."""
    print("### STEP 1: Preparing dataset (COCO → YOLO format)... ###")
    run_coco_to_yolo_conversion()

def step_distill():
    """STEP 2: Run knowledge distillation for both teacher models."""
    print("### STEP 2.1: Running Knowledge Distillation with ConvNeXt Teacher... ###")
    run_manual_ddp_script(str(config.SRC_DIR / "distillation/trainer_convnext.py"), cwd=config.ROOT_DIR)
    
    print("\n### STEP 2.2: Running Knowledge Distillation with ViT Teacher... ###")
    run_manual_ddp_script(str(config.SRC_DIR / "distillation/trainer_vit.py"), cwd=config.ROOT_DIR)

def step_finetune():
    """STEP 3: Run fine-tuning experiments for all models."""
    print("### STEP 3.1: Generating latest RT-DETR config files... ###")
    run_config_generation()
    
    rtdetr_train_script = str(config.RTDETR_SOURCE_DIR / "tools/train.py")

    print("\n### STEP 3.2: Fine-tuning RT-DETR (ConvNeXt Distilled)... ###")
    run_manual_ddp_script(f"{rtdetr_train_script} -c {config.RTDETR_FINETUNE_CONFIG_CONVNEXT} --use-amp --seed=0", cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.3: Fine-tuning RT-DETR (ViT Distilled)... ###")
    run_manual_ddp_script(f"{rtdetr_train_script} -c {config.RTDETR_FINETUNE_CONFIG_VIT} --use-amp --seed=0", cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.4: Fine-tuning RT-DETR (Baseline)... ###")
    run_manual_ddp_script(f"{rtdetr_train_script} -c {config.RTDETR_FINETUNE_CONFIG_BASELINE} --use-amp --seed=0", cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.5: Fine-tuning YOLO (Baseline)... ###")
    train_yolo_baseline()

def main():
    """Main entry point to orchestrate the full training pipeline."""
    print("### Initializing: Installing dependencies and cloning RT-DETR repo... ###")
    subprocess.run(f"{sys.executable} -m pip install -q -r requirements.txt", shell=True, check=True)
    if not config.RTDETR_SOURCE_DIR.exists():
        subprocess.run(f"git clone https://github.com/lyuwenyu/RT-DETR.git {config.RTDETR_SOURCE_DIR}", shell=True, check=True)
    
    parser = argparse.ArgumentParser(description="Master training orchestrator for the ML pipeline.")
    parser.add_argument('--all', action='store_true', help='Run all steps: prepare-data, distill, finetune.')
    parser.add_argument('--prepare-data', action='store_true', help='Run only the data preparation step.')
    parser.add_argument('--distill', action='store_true', help='Run only the knowledge distillation steps.')
    parser.add_argument('--finetune', action='store_true', help='Run only the fine-tuning experiments.')
    
    args = parser.parse_args()

    run_all = not any([args.prepare_data, args.distill, args.finetune]) or args.all

    if run_all or args.prepare_data:
        step_prepare_data()
    if run_all or args.distill:
        step_distill()
    if run_all or args.finetune:
        step_finetune()

    print("\n✅ All selected processes completed successfully.")

if __name__ == "__main__":
    main()