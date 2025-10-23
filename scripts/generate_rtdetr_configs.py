import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def _generate_config_file(template_filename: str, output_filename: str, replacements: dict):
    template_path = config.RTDETR_SOURCE_DIR / 'configs' / 'rtdetrv2' / template_filename
    output_path = config.CONFIG_GEN_DIR / output_filename

    config.CONFIG_GEN_DIR.mkdir(parents=True, exist_ok=True)

    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in replacements.items():
        replacement_value = str(Path(value).absolute()).replace('\\', '/')
        content = content.replace(f"{{{key}}}", replacement_value)

    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated config file: {output_path}")

def run_config_generation():
    print("--- Generating RT-DETR config files from templates... ---")
    
    common_replacements = {
        "TRAIN_IMG_FOLDER": config.COCO_TRAIN_IMAGES,
        "VAL_IMG_FOLDER": config.COCO_VAL_IMAGES,
        "TRAIN_ANN_FILE": config.COCO_TRAIN_ANNOTATIONS,
        "VAL_ANN_FILE": config.COCO_VAL_ANNOTATIONS,
    }

    convnext_checkpoint_path = config.RTDETR_SOURCE_DIR / config.CONVNEXT_BEST_WEIGHTS.name
    vit_checkpoint_path = config.RTDETR_SOURCE_DIR / config.VIT_CONVERTED_WEIGHTS.name

    # Config cho model ConvNeXt-distilled
    _generate_config_file(
        "rtdetrv2_taco_finetune_convnext.yml.template",
        "rtdetrv2_taco_finetune_convnext.yml",
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR / "rtdetrv2_finetune_taco_convnext_teacher",
            "TUNING_CHECKPOINT": convnext_checkpoint_path, 
        }
    )

    # Config cho model ViT-distilled
    _generate_config_file(
        "rtdetrv2_taco_finetune_vit.yml.template",
        "rtdetrv2_taco_finetune_vit.yml",
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR / "rtdetrv2_finetune_taco_vit_teacher",
            "TUNING_CHECKPOINT": vit_checkpoint_path, # Sửa ở đây
        }
    )
    
    # Config cho model Baseline
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