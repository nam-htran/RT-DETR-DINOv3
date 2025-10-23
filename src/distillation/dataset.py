import os
from PIL import Image
import torch
from pycocotools.coco import COCO

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
        # Trả về hình ảnh và một nhãn giả (0) vì DataLoader yêu cầu
        return self.transforms(img), 0

    def __len__(self):
        return len(self.ids)