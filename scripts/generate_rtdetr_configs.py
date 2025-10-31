# ===== scripts/generate_rtdetr_configs.py =====
import sys
from pathlib import Path
import yaml

# Add project root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def _generate_config_file(template_filename: str, output_filename: str, replacements: dict):
    """Generates a config file from a template inside the RTDETR config directory."""
    # Path to the template file inside the cloned rtdetr repo
    template_path = config.RTDETR_CONFIG_DIR / template_filename
    # Path to the output .yml file, also inside the rtdetr repo
    output_path = config.RTDETR_CONFIG_DIR / output_filename

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}. Please ensure it exists.")

    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in replacements.items():
        # Ensure paths are absolute and use forward slashes
        replacement_value = str(Path(value).absolute()).replace('\\', '/')
        content = content.replace(f"{{{key}}}", replacement_value)

    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated config file: {output_path}")

def run_config_generation():
    """Generates all necessary RT-DETR config files for fine-tuning."""
    print("--- Generating RT-DETR config files from templates... ---")
    
    common_replacements = {
        "TRAIN_IMG_FOLDER": config.COCO_TRAIN_IMAGES,
        "VAL_IMG_FOLDER": config.COCO_VAL_IMAGES,
        "TRAIN_ANN_FILE": config.COCO_TRAIN_ANNOTATIONS,
        "VAL_ANN_FILE": config.COCO_VAL_ANNOTATIONS,
    }

    # Config for ConvNeXt-distilled model
    _generate_config_file(
        "rtdetrv2_taco_finetune_convnext.yml.template",
        "rtdetrv2_taco_finetune_convnext.yml",
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR / "rtdetrv2_finetune_taco_convnext_teacher",
            "TUNING_CHECKPOINT": config.CONVNEXT_BEST_WEIGHTS,
        }
    )

    # Config for ViT-distilled model
    _generate_config_file(
        "rtdetrv2_taco_finetune_vit.yml.template",
        "rtdetrv2_taco_finetune_vit.yml",
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR / "rtdetrv2_finetune_taco_vit_teacher",
            "TUNING_CHECKPOINT": config.VIT_BEST_WEIGHTS,
        }
    )
    
    # Config for Baseline model
    _generate_config_file(
        "rtdetrv2_taco_finetune_BASELINE.yml.template",
        "rtdetrv2_taco_finetune_BASELINE.yml",
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_BASELINE_OUTPUT_DIR / "rtdetrv2_finetune_taco_BASELINE",
            "TUNING_CHECKPOINT": "''"
        }
    )
    print("--- Config generation complete. ---")

if __name__ == "__main__":
    run_config_generation()