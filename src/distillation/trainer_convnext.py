import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from tqdm import tqdm
import wandb
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config as project_config


class HuggingFaceTeacherWrapper(nn.Module):
    def __init__(self, model_id: str, token: str = None):
        super().__init__()
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Loading teacher model '{model_id}' from Hugging Face...")
        config = AutoConfig.from_pretrained(model_id, token=token)
        self._model = AutoModel.from_pretrained(model_id, token=token)
        self.is_vit = "vit" in config.model_type.lower()
        self._feature_dim = (
            self._model.config.hidden_size
            if self.is_vit
            else self._model.config.hidden_sizes[-1]
        )
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Detected {'ViT' if self.is_vit else 'ConvNeXT'} architecture. Feature dim: {self._feature_dim}")

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._model(pixel_values=x, output_hidden_states=True)
        if self.is_vit:
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            b, s, d = patch_tokens.shape
            h = w = int(math.sqrt(s))
            return patch_tokens.permute(0, 2, 1).reshape(b, d, h, w)
        return outputs.hidden_states[-1]

class CocoDetectionForDistill(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transforms):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        return self.transforms(img), 0

    def __len__(self):
        return len(self.ids)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    if os.environ.get("RANK", "0") == "0":
        print(f"[Rank {os.environ.get('RANK')}] Process group initialized successfully.")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main_training_function(rank, world_size, config):
    device = rank
    is_main_process = (rank == 0)
    
    hf_token = None
    if is_main_process:
        print(f"Running DDP on {world_size} GPUs.")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"run_ddp_{timestamp}_lr{config['learning_rate']}_bs{config['batch_size_per_gpu']}"
        try:
            from huggingface_hub import login
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            wandb_key = os.getenv("WANDB_API_KEY")
            if hf_token: login(token=hf_token)
            if wandb_key: wandb.login(key=wandb_key)
            wandb.init(project=config["wandb_project"], config=config, name=run_name)
        except Exception as e:
            print(f"Could not log in to Hugging Face or W&B, continuing without them. Error: {e}")

        best_weights_path = Path(config["best_weights_filename"])
        final_weights_path = Path(config["final_weights_filename"])
        
        best_weights_path.parent.mkdir(parents=True, exist_ok=True)
        final_weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {best_weights_path.parent}")
    
    dist.barrier()
    
    teacher_model = HuggingFaceTeacherWrapper(config["teacher_hf_id"], token=hf_token).to(device)
    teacher_model.eval()

    if is_main_process:
        rtdetr_local_path = str(project_config.ROOT_DIR / 'rtdetr')
        print(f"Loading student model from local path: {rtdetr_local_path}")
        torch.hub.load(rtdetr_local_path, "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)

    dist.barrier() 

    rtdetr_local_path = str(project_config.ROOT_DIR / 'rtdetr')
    student_hub_model = torch.hub.load(rtdetr_local_path, "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    student_model = student_hub_model.model.to(device)

    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640).to(device)
        student_features = student_model.encoder(student_model.backbone(x))
        student_channels = student_features[-1].shape[1]
    teacher_channels = teacher_model.feature_dim()
    projection_layer = nn.Conv2d(student_channels, teacher_channels, kernel_size=1).to(device)

    student_model = DDP(student_model, device_ids=[device], find_unused_parameters=True)
    projection_layer = DDP(projection_layer, device_ids=[device], find_unused_parameters=True)
    
    transforms = T.Compose([
        T.Resize((640, 640)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CocoDetectionForDistill(
        root=config["train_images_dir"],
        ann_file=config["train_ann_file"],
        transforms=transforms
    )
    val_dataset = CocoDetectionForDistill(
        root=config["val_images_dir"],
        ann_file=config["val_ann_file"],
        transforms=transforms
    )
    if is_main_process:
        print(f"Data loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size_per_gpu"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True, drop_last=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size_per_gpu"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True, drop_last=False, sampler=val_sampler
    )

    params = list(student_model.module.backbone.parameters()) + \
             list(student_model.module.encoder.parameters()) + \
             list(projection_layer.module.parameters())
             
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_factor'], patience=config['scheduler_patience'])

    if is_main_process and wandb.run:
        wandb.watch((student_model, projection_layer), log="all", log_freq=100)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    if is_main_process:
        print("Starting training...")
        
    for epoch in range(config["epochs"]):
        train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        student_model.train()
        projection_layer.train()
        total_train_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", disable=not is_main_process)

        for images, _ in train_iterator:
            images = images.to(device)
            with torch.no_grad():
                teacher_features = teacher_model(images)
            
            student_features = student_model.module.encoder(student_model.module.backbone(images))[-1]
            projected = projection_layer(student_features)
            teacher_resized = F.interpolate(teacher_features, size=projected.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = criterion(projected, teacher_resized)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        train_loss_tensor = torch.tensor(total_train_loss).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)

        student_model.eval()
        projection_layer.eval()
        total_val_loss = 0.0
        
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", disable=not is_main_process)
        with torch.no_grad():
            for images, _ in val_iterator:
                images = images.to(device)
                teacher_features = teacher_model(images)
                student_features = student_model.module.encoder(student_model.module.backbone(images))[-1]
                projected = projection_layer(student_features)
                teacher_resized = F.interpolate(teacher_features, size=projected.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(projected, teacher_resized)
                total_val_loss += loss.item()
                
        val_loss_tensor = torch.tensor(total_val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        
        if is_main_process:
            duration = time.time() - start_time
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {duration:.2f}s")
            if wandb.run:
                wandb.log({
                    "epoch": epoch + 1, 
                    "train/avg_loss": avg_train_loss, 
                    "val/avg_loss": avg_val_loss, 
                    "time/epoch_s": duration, 
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model...")
                # We need to access the underlying model from the DDP wrapper
                best_weights = {
                    **student_model.module.backbone.state_dict(), 
                    **student_model.module.encoder.state_dict()
                }
                torch.save({'model': best_weights}, config["best_weights_filename"])
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{config['early_stopping_patience']}")

        stop_training_tensor = torch.tensor(early_stopping_counter, device=device)
        dist.broadcast(stop_training_tensor, src=0)
        
        if stop_training_tensor.item() >= config['early_stopping_patience']:
            if is_main_process:
                print("Early stopping triggered. Finishing training.")
            break
            
    if is_main_process:
        print("\nDistillation finished.")
        final_weights = {
            **student_model.module.backbone.state_dict(), 
            **student_model.module.encoder.state_dict()
        }
        torch.save({'model': final_weights}, config["final_weights_filename"])
        print(f"Saved final epoch weights to '{config['final_weights_filename']}'")
        print(f"Best weights were saved to '{config['best_weights_filename']}' with val_loss: {best_val_loss:.4f}")
        if wandb.run:
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.finish()


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv(project_config.ROOT_DIR / '.env')

    try:
        setup_ddp()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        training_config = {
            "learning_rate": 1e-4, 
            "epochs": 50, 
            "batch_size_per_gpu": 2,
            "num_workers": 2, 
            "weight_decay": 1e-5,
            "teacher_hf_id": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            
            "train_images_dir": str(project_config.COCO_TRAIN_IMAGES),
            "val_images_dir": str(project_config.COCO_VAL_IMAGES),
            "train_ann_file": str(project_config.COCO_TRAIN_ANNOTATIONS),
            "val_ann_file": str(project_config.COCO_VAL_ANNOTATIONS),

            "scheduler_patience": 3, 
            "scheduler_factor": 0.1,
            "early_stopping_patience": 7,
            
            "best_weights_filename": str(project_config.CONVNEXT_BEST_WEIGHTS),
            "final_weights_filename": str(project_config.CONVNEXT_FINAL_WEIGHTS),
            "wandb_project": project_config.WANDB_PROJECT_CONVNEXT_DISTILL,
        }
        
        main_training_function(rank, world_size, training_config)
    finally:
        cleanup_ddp()