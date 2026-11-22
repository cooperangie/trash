# Waste Recognition Algorithm

University project for the **Computer Vision** course (A.Y. 2025/26) — Department of Mathematics, Computer Science and Physics, University of Udine.

- **Author:** Angela Rossi
- **Professor:** Prof. Christian Micheloni

---

## Overview and Objectives

This project addresses the challenge of automatic waste classification for recycling in scenarios with limited labeled data, analyzing the trade-off between accuracy, generalization in real-world environments, and energy impact (Green AI).

The study compares three main aspects:
1. **Transfer Learning:** Fine-tuning a lightweight pre-trained convolutional neural network (MobileNetV2) on a controlled dataset (TrashNet).
2. **Zero-Shot Learning:** Using the multimodal CLIP (ViT-B/32) model without any retraining, evaluated first on the 6 standard TrashNet classes and then extended to a 58-class open-world vocabulary integrating complex real-world categories (TACO).
3. **Measured Environmental Impact:** Real-time monitoring of energy consumption and $\text{CO}_2\text{eq}$ emissions during training and inference phases using the CodeCarbon library.

---

## Datasets

| Feature | TrashNet | TACO (Trash Annotations in Context) |
| :--- | :--- | :--- |
| **Origin** | Stanford CS229 (2016) | Crowdsourced (Proença & Simões, 2020) |
| **Total Images** | 2,527 (512x384 px resolution) | 1,500 (4,784 total annotations) |
| **Classes** | 6 materials (`glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`) | 60 fine-grained categories (hierarchical taxonomy) |
| **Context & Background** | Neutral white background, controlled lighting, single object | Real-world cluttered environments (streets, beaches, woods), multiple objects |
| **Project Usage** | 70/15/15 split (1,768 train, 379 val, 380 test) for training and evaluation | Sample of 500 instances (381 unique images) for open-world zero-shot testing |

---

## Methodology and Architecture

### 1. Preprocessing and Data Augmentation
- **TrashNet:** Resized to `224x224` px and normalized using ImageNet mean and standard deviation. Applied *Random Resized Crop* (80%-100%), horizontal flip, rotations up to 15 degrees, and *Color Jitter* on the training set.
- **TACO:** Isolated individual instances by cropping bounding boxes with a 15% context margin, incorporating a scaling correction factor to align original coordinates with the downloaded resized images (`flickr_640_url`).
- **Real-World Inference:** Implemented a proportional resize pipeline followed by a *center crop* to `224x224` px to prevent geometric distortion on non-native 4:3 photos.

### 2. Models Compared
- **MobileNetV2 (Transfer Learning):** ImageNet pre-trained backbone completely frozen; retrained only the final classifier head on the 6 TrashNet classes (7,686 trainable parameters out of 2,231,558 total, or 0.3%). Adam optimizer (`lr=1e-3`), batch size 32, 10 epochs.
- **CLIP ViT-B/32 (Zero-Shot):** Approximately 151.3 million parameters. Classification based on cosine similarity between the image embedding and text prompts of the target classes. For the open-world benchmark, a dynamic vocabulary of 58 classes was built (6 TrashNet + 4 manually added categories: battery, electronic, organic, textile + sampled TACO categories).

### 3. Emissions Tracking
- Integrated **CodeCarbon** (`EmissionsTracker`) to measure actual CPU, GPU, and RAM power consumption and compute grams of $\text{CO}_2$ equivalent emitted based on the carbon intensity of the underlying data center grid.

---

## Experimental Results

| Configuration | Dataset | Accuracy | Training Emissions | Inference Emissions | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Transfer Learning (MobileNetV2)** | TrashNet (6 classes) | **81.8%** | 0.82 g $\text{CO}_2\text{eq}$ | 0.0079 g $\text{CO}_2\text{eq}$ | 1.45 s |
| **Zero-Shot CLIP (ViT-B/32)** | TrashNet (6 classes) | 56.1% | 0 (no training) | 0.0496 g $\text{CO}_2\text{eq}$ | 8.56 s |
| **Zero-Shot CLIP Open-World** | TrashNet + TACO (58 classes) | 12.0% | 0 (no training) | 0.1885 g $\text{CO}_2\text{eq}$ | 38.54 s |

### Results Analysis
- **Specialization vs. Generalization:** MobileNetV2 achieves higher accuracy on the known domain (81.8%) with minimal inference time and emissions. CLIP demonstrates high flexibility without requiring retraining, but suffers from semantic overlap across large vocabularies (58 classes) and tends to confuse clear plastic with glass (`glass` recall 0.85 vs. `plastic` recall 0.03 on TrashNet).
- **Real-World Testing:** On an out-of-distribution real-world photograph (plastic bottles in a trash bin), the fine-tuned MobileNetV2 model with *center-crop* preprocessing accurately classified the subject as `plastic` with **92.8%** confidence.
- **Energy Balance:** Although CLIP generates zero local training emissions (as its computational cost was incurred upstream during pre-training), its inference phase is significantly more resource-intensive than a specialized lightweight CNN.

---

## Requirements and Installation

This project was developed in Python (Google Colab / Jupyter Notebook environment). Main dependencies required:

```bash
pip install torch torchvision transformers datasets codecarbon pillow matplotlib seaborn scikit-learn
