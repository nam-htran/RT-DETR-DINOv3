# RT-DisDINO: Knowledge Distillation for RT-DETR with DINOv2 Teachers

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=for-the-badge)![WandB](https://img.shields.io/badge/Weights%20&%20Biases-FFCC00?style=for-the-badge&logo=weightsandbiases&logoColor=black)![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey?style=for-the-badge)

This repository presents a complete pipeline for enhancing the **RT-DETR-L** object detection model using **Knowledge Distillation (KD)**. We employ two powerful "teacher" models from the DINOv2 family, **ConvNeXt-Base** and **ViT-Base**, to transfer their rich feature representations to the "student" model, RT-DETR. The entire training, distillation, and evaluation process is performed on the **TACO** waste dataset.

This project is a professional refactoring of an initial, stable [Kaggle notebook](https://www.kaggle.com/code/tranhoangnamk18hcm/rt-disdinov2-distillation-finetune-analysis), designed to create a structured, reproducible, and easily extensible workflow for experimentation.

## Project Workflow

The diagram below illustrates the complete experimental pipeline, from data preparation to final model comparison.

## Key Features
- **End-to-End Knowledge Distillation:** A complete workflow for feature-based distillation from Hugging Face teachers to an RT-DETR student.
- **Comprehensive Benchmarking:** Compares the distilled models against both the RT-DETR baseline and a strong YOLOv11 baseline.
- **Reproducible Structure:** Refactored from a notebook into a modular project with centralized configuration (`config.py`) and an orchestration script (`train.py`).
- **Hugging Face Hub Integration:** Pre-trained distilled models are publicly available on the Hub for immediate use.
- **Experiment Tracking:** Integrated with Weights & Biases for logging metrics, comparing runs, and visualizing results.

## Results and Analysis

The final models were benchmarked on a **Tesla T4 GPU**. The results below summarize the performance after fine-tuning on the TACO dataset.

| Model                         | mAP@50-95 | mAP@50 | Speed (ms) | Params (M) | FLOPs (G) |
| ----------------------------- | :-------: | :----: | :--------: | :--------: | :-------: |
| RT-DETR-L (Baseline)          |   3.10%   | 4.00%  |   59.94    |   40.92    |  136.06   |
| YOLOv11-L (Baseline)          |  22.94%   | 26.37% | **29.42**  | **25.36**  | **87.53** |
| **RT-DisDINOv2 (ConvNeXt)**   | **6.00%** | **8.20%** |   58.76    |   40.92    |  136.06   |
| RT-DisDINOv2 (ViT)            |   2.90%   | 4.00%  |   59.04    |   40.92    |  136.06   |

### Key Findings
1.  **ConvNeXt Teacher Succeeds:** Knowledge distillation from the **DINOv3 ConvNeXt-Base** teacher provided a significant performance boost, nearly **doubling the mAP@50-95** (3.10% -> 6.00%) compared to the RT-DETR baseline.
2.  **ViT Teacher Struggles:** Distillation from the **DINOv3 ViT-Base** teacher failed to improve performance and resulted in slightly lower metrics than the baseline.
3.  **YOLOv11 Dominates:** The YOLOv11-L baseline significantly outperforms all RT-DETR variants in both accuracy and efficiency (speed, parameters, FLOPs) on this dataset.

### Algorithmic Analysis

#### Why did ConvNeXt succeed?
The success of the ConvNeXt teacher can be attributed to **architectural similarity and feature compatibility**.
- **Shared Inductive Bias:** Both the teacher (ConvNeXt) and the student's backbone (a ResNet-variant) are Convolutional Neural Networks (CNNs). They share fundamental inductive biases like **locality** and **translation equivariance**. This means their feature maps are spatially structured in a similar way, making it easier for the student to mimic the teacher's representations.
- **Multi-level Feature Distillation:** Our distillation process utilized feature maps from multiple intermediate layers of the ConvNeXt teacher. This provided the student with a rich, hierarchical understanding of features, from low-level textures to more complex semantic information, which is highly beneficial for object detection.

#### Why did ViT struggle?
The performance drop with the ViT teacher likely stems from a significant **architectural mismatch**.
- **Differing Inductive Biases:** Vision Transformers (ViTs) operate on the principle of **global self-attention**, processing images as a sequence of patches. This is fundamentally different from the local, hierarchical processing of CNNs. The feature space of a ViT is less spatially explicit than that of a CNN, which can make direct feature map distillation challenging.
- **Representation Gap:** We distilled from the final hidden state of the ViT, which represents patch-level features after global context aggregation. This high-level, abstract representation may not be directly compatible with the spatially-grounded feature maps expected by the RT-DETR's CNN backbone and transformer encoder. The student might struggle to translate the ViT's global context into the local feature patterns it is built to understand.

## Setup and Installation
It is recommended to use **WSL/Linux** due to its stability with PyTorch's distributed training.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/nam-htran/RT-DisDINOV3
    cd RT-DisDINOV3
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n rtdisdino python=3.11 -y
    conda activate rtdisdino
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    Download the TACO dataset from this Kaggle link: [TACO Dataset for Object Detection](https://www.kaggle.com/datasets/tranhoangnamk18hcm/dsp-pre-final). Extract the archive and place its contents into `data_input/processed_taco_coco/`, ensuring the final structure is:
    ```
    data_input/
    └── processed_taco_coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
    ```

5.  **Set Up Environment Variables:**
    Create a `.env` file from the example and add your API keys.
    ```bash
    cp .env.example .env
    ```
    Open `.env` and fill in your credentials:
    ```.env
    HUGGINGFACE_TOKEN="hf_..."
    WANDB_API_KEY="..."
    ```

## Training Pipeline

The `train.py` script is the central orchestrator for all experiments.

### 1. Run the Full Pipeline (Recommended)
This command executes all steps sequentially: data preparation, distillation of both models, and fine-tuning of all four variants.
```bash
python train.py --all
```

### 2. Run Individual Steps
You can also run each part of the pipeline independently.

*   **Step 1: Prepare Data (COCO to YOLO conversion for baseline)**
    ```bash
    python train.py --prepare-data
    ```
*   **Step 2: Knowledge Distillation**
    ```bash
    python train.py --distill
    ```
*   **Step 3: Fine-tuning (All models)**
    ```bash
    python train.py --finetune
    ```

## Project Structure
```
.
├── config.py                   # Central hub for all paths and configurations
├── train.py                    # Main script to orchestrate the entire pipeline
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup for installing the 'src' package
├── rtdetr/                     # Git submodule containing the original RT-DETR source code
├── scripts/                    # Helper scripts for data conversion and config generation
│   ├── convert_coco_to_yolo.py
│   └── generate_rtdetr_configs.py
└── src/                        # Main project source code
    ├── distillation/           # Logic for the knowledge distillation process
    │   ├── trainer_convnext.py # DDP training script for ConvNeXt teacher
    │   └── trainer_vit.py      # DDP training script for ViT teacher
    └── finetune/               # Logic for the fine-tuning process
        ├── trainer_rtdetr.py   # (Legacy) Orchestrator for RT-DETR fine-tuning
        └── trainer_yolo.py     # Script to fine-tune the YOLO baseline
```

## Kaggle Notebooks
The `/kaggle` directory contains the original, stable notebooks where this research was initially conducted:
*   `rt-dinov3.ipynb`: The complete, self-contained notebook to execute the entire training and distillation pipeline on Kaggle's platform.
*   `aftertrain-analysis.ipynb`: A comprehensive notebook for analyzing, evaluating, and comparing the results of the trained models, including generating plots and benchmark tables.

## License
This project is licensed under the Apache 2.0 License. It incorporates components with their own licenses:
- The core RT-DETR code is licensed under **Apache 2.0**.
- The DINOv2 teacher models are governed by the **Custom DINOv3 License**.

Please review the respective license files for full details.

## Acknowledgements
This work builds upon the fantastic open-source contributions from the community:
*   **RT-DETR**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
*   **DINOv2**: Meta AI for their powerful foundation models.
*   **TACO Dataset**: [pedropro/TACO](https://github.com/pedropro/TACO) and the Kaggle community for maintaining accessible versions.
*   **Hugging Face**: For their `transformers` library and model hub, which greatly simplified teacher model integration.
