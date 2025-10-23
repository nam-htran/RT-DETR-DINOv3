# RT-DisDINOv3: Knowledge Distillation for RT-DETR using DINOv3 Teachers

This project focuses on enhancing the performance of the **RT-DETR-L** object detection model through **Knowledge Distillation**. We leverage two powerful "teacher" models from the DINOv3 family, **ConvNeXt-Base** and **ViT-Base**, to transfer knowledge to the "student" model, RT-DETR. The entire training and evaluation process is conducted on the **TACO** waste dataset.

This project is a refactored version of an initial Kaggle notebook, aiming for a structured, reproducible, and professional workflow.

## Pre-trained Models (RT-DisDINOv3)

The two RT-DETR models distilled with TACO dataset through this project are publicly available on the Hugging Face Hub. You can easily download and use them.

| Model Name                  | Teacher Model        | Hugging Face Hub Link                                                     |
| :-------------------------- | :------------------- | :------------------------------------------------------------------------ |
| **RT-DisDINOv3-ConvNext**   | DINOv3 ConvNeXt-Base | [`hnamt/RT-DisDINOv3-ConvNext-Base`](https://huggingface.co/hnamt/RT-DisDINOv3-ConvNext-Base) |
| **RT-DisDINOv3-ViT**        | DINOv3 ViT-Base      | [`hnamt/RT-DisDINOv3-ViT-Base`](https://huggingface.co/hnamt/RT-DisDINOv3-ViT-Base)       |

### How to Use

You can load these distilled weights and apply them to the original RT-DETR-L model from `torch.hub`.

```python
import torch
from torch.hub import load_state_dict_from_url

# 1. Load the original RT-DETR-L model
rtdetr_l = torch.hub.load('lyuwenyu/RT-DETR', 'rtdetrv2_l', pretrained=True)
model = rtdetr_l.model

# 2. Choose and load the distilled weights from the Hugging Face Hub
# Change the URL to select the model you want
# MODEL_URL = "https://huggingface.co/hnamt/RT-DisDINOv3-ConvNext-Base/resolve/main/distilled_rtdetr_convnext_teacher_BEST.pth"
MODEL_URL = "https://huggingface.co/hnamt/RT-DisDINOv3-ViT-Base/resolve/main/distilled_rtdetr_vit_teacher_BEST.pth"

# Load the state_dict
distilled_state_dict = load_state_dict_from_url(MODEL_URL, map_location='cpu')['model']

# 3. Load the weights into the model's backbone and encoder
model.load_state_dict(distilled_state_dict, strict=False)

print("Successfully loaded and applied distilled knowledge weights!")

# Now you can use the 'model' for fine-tuning on your own dataset
# or for inference.
# model.eval()
# ...
```

## Project Structure

The project is organized to be easily manageable and extensible:
```
.
├── config.py                   # Central management for all paths and configurations
├── requirements.txt            # Required Python libraries
├── train.py                    # Main script to orchestrate the entire pipeline
├── rtdetr/                     # Submodule containing the original RT-DETR source code
├── scripts/                    # Helper scripts (data conversion, config generation)
│   ├── convert_coco_to_yolo.py
│   ├── convert_lightly_checkpoint.py
│   └── generate_rtdetr_configs.py
├── src/                        # Main project source code
│   ├── distillation/           # Source code for the knowledge distillation process
│   │   ├── trainer_convnext.py
│   │   └── trainer_vit.py
│   └── finetune/               # Source code for the fine-tuning process
│       ├── trainer_rtdetr.py
│       └── trainer_yolo.py
└── output/                     # Directory for outputs (weights, logs)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nam-htran/RT-DisDINOV3
    cd RT-DisDINOV3
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n rtdis python=3.10
    conda activate rtdis
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data:**
    Download the TACO dataset from Kaggle: [TACO Dataset for Object Detection](https://www.kaggle.com/datasets/tranhoangnamk18hcm/dsp-pre-final).
    Extract it and place it into the `data_input/processed_taco_coco` directory, ensuring the structure is as follows:
    ```
    data_input/
    └── processed_taco_coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
    ```

5.  **Set up environment variables:**
    Copy the `.env.example` file to `.env` and fill in your API keys for Weights & Biases and Hugging Face Hub.
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your credentials:
    ```.env
    HUGGINGFACE_TOKEN="hf_..."
    WANDB_API_KEY="..."
    DINOV3_TEACHER_URL="..."
    ```

## Training Pipeline

The `train.py` script is the main entry point for running experiments.

### 1. Run the Full Pipeline
To execute all steps from data preparation to fine-tuning, use the following command:
```bash
python train.py --all
```

### 2. Run Individual Steps
*   **Step 1: Prepare Data**
    ```bash
    python train.py --prepare-data
    ```
*   **Step 2: Knowledge Distillation**
    ```bash
    python train.py --distill
    ```
*   **Step 3: Convert Checkpoint (for ViT)**
    ```bash
    python train.py --convert
    ```
*   **Step 4: Fine-tuning**
    ```bash
    python train.py --finetune
    ```

## Note on Kaggle Notebooks
The `/kaggle` directory contains two notebooks designed to run smoothly in the Kaggle environment:
*   `rt-dinov3.ipynb`: A complete notebook to execute the entire training pipeline on Kaggle.
*   `aftertrain-analysis.ipynb`: A notebook for analyzing, evaluating, and comparing the results of the trained models.

## Results and Analysis
The table below summarizes the performance of the models after fine-tuning on the TACO dataset, benchmarked on a Tesla T4 GPU.

| Model                         | mAP@50-95 | mAP@50 | Speed (ms) | Params (M) | FLOPs (G) | Notes                               |
| ----------------------------- | :-------: | :----: | :--------: | :--------: | :-------: | ----------------------------------- |
| RT-DETR-L (Baseline)          |   2.80%   | 4.60%  |   50.05    |   30.95    |  109.95   | Fine-tuned from pre-trained COCO.   |
| YOLOv11-L (Baseline)          |  22.94%   | 26.37% | **31.72**  | **25.36**  | **87.53** | Superior performance over RT-DETR. |
| **RT-DisDINOv3 (w/ ConvNeXt)** | **3.60%** | **5.30%** |   49.80    |   30.95    |  109.95   | +28.6% mAP increase over baseline. |
| RT-DisDINOv3 (w/ ViT)         |   2.80%   | 4.20%  |   49.80    |   30.95    |  109.95   | No performance improvement.

Detailed results and analysis charts can be tracked directly on **Weights & Biases**.

## License
This project is licensed under the Apache 2.0 License. However, it incorporates components with their own licenses.
- The core RT-DETR code is licensed under Apache 2.0.
- The DINOv3 teacher models are governed by the DINOv3 License.

Please see the `LICENSE` file for more details. We recommend creating a `LICENSES` directory and placing the full text of the DINOv3 and Apache 2.0 licenses there.

## Acknowledgements
This project is built upon the excellent open-source work from the community:
*   **RT-DETR**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
*   **DINOv3**: Thanks to Meta AI for releasing powerful foundation models.
*   **TACO Dataset**: [pedropro/TACO](https://github.com/pedropro/TACO)
*   **Lightly**: The [lightly-train](https://www.lightly.ai/) library simplified the knowledge distillation process.
