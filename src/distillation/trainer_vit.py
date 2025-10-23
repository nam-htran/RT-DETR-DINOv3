import os
import sys
import shutil
import torch
import wandb
import lightly_train
from lightly_train.model_wrappers import RTDETRModelWrapper
import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config as project_config

def main_training_function(config):
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

    if is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"run_ddp_{timestamp}_lr{config['learning_rate']}_bs{config['batch_size_per_gpu']}"
        try:
            wandb_key = os.getenv("WANDB_API_KEY")
            wandb.login(key=wandb_key)
        except Exception as e:
            print(f"W&B secrets not available. Skipping login. Error: {e}")

        if os.path.exists(config['output_dir']):
            print(f"Output directory '{config['output_dir']}' already exists. Deleting it.")
            shutil.rmtree(config['output_dir'])

    if not is_main_process:
        run_name = ""

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process:
        print("Initializing Student Model (RT-DETR)...")
    rtdetr_local_path = str(project_config.ROOT_DIR / 'rtdetr')
    student_hub_model = torch.hub.load(rtdetr_local_path, "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    wrapped_student = RTDETRModelWrapper(student_hub_model.model)
    
    callbacks_config = {
        "model_checkpoint": {
            "dirpath": os.path.join(config['output_dir'], 'checkpoints'),
            "filename": 'best-model-{epoch}-{validation_loss:.4f}',
            "monitor": 'val_loss',
            "mode": 'min',
            "save_top_k": 1,
        },
        "learning_rate_monitor": {}
    }
    
    global_batch_size = config['batch_size_per_gpu'] * config['num_gpus']

    if is_main_process:
        print("Starting distillation with lightly_train.train()...")
        print(f"Global batch size: {global_batch_size} ({config['batch_size_per_gpu']} per GPU)")

    lightly_train.train(
        model=wrapped_student,
        method="distillationv1",
        method_args={
            "teacher": config['teacher_name'],
            "teacher_url": config['teacher_url'],
        },
        data=[config['train_dir'], config['val_dir']],
        out=config['output_dir'],
        epochs=config['epochs'],
        batch_size=global_batch_size,
        num_workers=config['num_workers'],
        optim=config['optimizer_name'],
        optim_args={"lr": config['learning_rate'], "weight_decay": config['weight_decay']},
        callbacks=callbacks_config,
        loggers={
            "wandb": {
                "project": config['wandb_project'],
                "name": run_name,
            }
        },
        devices=config['num_gpus'],
        strategy='ddp_find_unused_parameters_true',
        accelerator='gpu'
    )
    if is_main_process:
        print("\nDistillation finished.")
        print(f"Best model checkpoint saved in directory: {os.path.join(config['output_dir'], 'checkpoints')}")
        
if __name__ == '__main__':
    DINOV3_VIT_TEACHER_URL = os.getenv("DINOV3_TEACHER_URL")
    
    if not DINOV3_VIT_TEACHER_URL:
        print("ERROR: Save your token key into kaggle secret")
    else:
        training_config = {
            "num_gpus": project_config.NUM_GPUS_PER_NODE,
            "epochs": 50, "batch_size_per_gpu": 8, "num_workers": 2,
            "optimizer_name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-5,
            "early_stopping_patience": 7,
            "teacher_name": "dinov3/vitb16", 
            "teacher_url": DINOV3_VIT_TEACHER_URL,
            "train_dir": str(project_config.COCO_TRAIN_IMAGES),
            "val_dir": str(project_config.COCO_VAL_IMAGES),
            "output_dir": str(project_config.VIT_DISTILL_OUTPUT_DIR),
            "wandb_project": project_config.WANDB_PROJECT_VIT_DISTILL
        }
        
        main_training_function(training_config)