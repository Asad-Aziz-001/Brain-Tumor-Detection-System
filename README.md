---
tags:
- image-classification
- medical-imaging
- brain-tumor
- mobilenet
- grad-cam
- pytorch
- explainable-ai
license: apache-2.0
datasets:
- Hemg/brain-tumor-classification-mri
metrics:
- accuracy
- f1
---

# Brain Tumor Detection with Grad-CAM Heatmaps

A medical image classification model for detecting brain tumors from MRI scans, built with **MobileNetV2** transfer learning and **Grad-CAM** explainability visualizations.

## Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **93.57%** |
| Macro F1-Score | 0.94 |
| Architecture | MobileNetV2 (ImageNet V2 pretrained) |
| Dataset | [Hemg/brain-tumor-classification-mri](https://huggingface.co/datasets/Hemg/brain-tumor-classification-mri) |
| Classes | glioma, meningioma, no tumor, pituitary |
| Training Epochs | 8 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma Tumor | 0.95 | 0.89 | 0.92 |
| Meningioma Tumor | 0.90 | 0.92 | 0.91 |
| No Tumor | 0.92 | 0.97 | 0.94 |
| Pituitary Tumor | 0.96 | 0.97 | 0.97 |

## Training Recipe

Based on published medical imaging research:
- **[arxiv:2506.09161](https://arxiv.org/abs/2506.09161)**: MobileNetV2 for brain tumor MRI classification
- **[arxiv:2307.10506](https://arxiv.org/abs/2307.10506)**: Two-phase fine-tuning (freeze then unfreeze)
- **[arxiv:1610.02391](https://arxiv.org/abs/1610.02391)**: Grad-CAM algorithm for CNN explainability

### Key Training Details
- **Phase 1** (epochs 1-3): Freeze early layers, train classifier + later conv layers
- **Phase 2** (epochs 4-8): Unfreeze all layers, fine-tune with lower LR + cosine annealing
- Adam optimizer, LR=3e-4, weight decay=1e-4
- Data augmentation: flip, rotation, color jitter
- Input: 160x160 RGB, ImageNet normalization

## Grad-CAM Heatmaps

Visualizations show which brain regions the model focuses on when making predictions.
Target layer:  (last convolutional block of MobileNetV2).

### Grand Heatmap Grid - All Tumor Types
Shows Original MRI, Grad-CAM activation, and Overlay for 3 samples per class:
![Grand Heatmap Grid](heatmaps/grand_heatmap_grid.png)

### Average Activation by Tumor Type
![Average Heatmaps](heatmaps/average_heatmaps_by_class.png)

### Glioma Tumor Analysis
![Glioma](heatmaps/class_glioma_tumor_strip.png)

### Meningioma Tumor Analysis
![Meningioma](heatmaps/class_meningioma_tumor_strip.png)

### No Tumor (Healthy) Analysis
![No Tumor](heatmaps/class_no_tumor_strip.png)

### Pituitary Tumor Analysis
![Pituitary](heatmaps/class_pituitary_tumor_strip.png)

### Training Curves
![Training History](training_history.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## Usage



## Files
- `best_model.pth` - PyTorch model weights
- `config.json` - Model configuration
- `training_history.json` - Per-epoch training metrics
- `confusion_matrix.png` - Test confusion matrix
- `training_history.png` - Loss/accuracy/LR curves
- `heatmaps/` - All Grad-CAM visualizations
