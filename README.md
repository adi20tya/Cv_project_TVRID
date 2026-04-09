# 🔍 TVRID 2026 – Privacy-Preserving Person Re-Identification

This repository contains my implementation for the **ICPR 2026 TVRID Challenge**, which focuses on **top-view person re-identification using RGB and Depth data**.

---

## 📌 Problem Overview

Traditional person re-identification relies heavily on facial features and frontal views.
However, this challenge introduces a **privacy-preserving setting**:

* Top-view (overhead cameras)
* Minimal facial visibility
* Multi-modal input:

  * RGB (appearance)
  * Depth (3D body structure)

### 🎯 Goal:

Given a query person, retrieve the **same identity across different cameras and conditions**.

---

## ⚙️ Dataset Characteristics

* 88 unique identities
* Captured using Intel RealSense D455 cameras
* Each person appears:

  * IN / OUT passages
  * Multiple viewpoints:

    * Flat ground
    * Ascending stairs
    * Descending stairs
    * Oblique roof view

---

## 🧠 Approach

### 🔹 Model Architecture

* Dual-stream architecture:

  * RGB Encoder → EfficientNetV2-L
  * Depth Encoder → EfficientNetV2-L
* Feature aggregation using **GeM pooling**
* Fusion:

  * Concatenation of RGB + Depth features
* Embedding dimension: 512

---

### 🔹 Training Strategy (Final Version)

This project follows a **metric learning approach**:

* **CrossEntropy Loss** → Identity classification
* **Triplet Loss** → Embedding learning (core for Re-ID)

👉 This hybrid approach ensures:

* Discriminative identity learning
* Structured embedding space

---

### 🔹 Key Techniques Used

* PK Sampling (multiple samples per identity per batch)
* Embedding normalization (L2 normalization)
* Dual-modal fusion (RGB + Depth)
* BNNeck (for better generalization)

---

## 📊 Evaluation Metrics

* CMC@1 / CMC@5 / CMC@10
* mAP (Mean Average Precision)

---

## 🚀 Improvements Over Initial Version

| Component      | Initial Version     | Improved Version               |
| -------------- | ------------------- | ------------------------------ |
| Objective      | Classification only | Metric Learning (Triplet + CE) |
| Sampling       | Random              | PK Sampling                    |
| Depth Handling | Treated as RGB      | Proper depth processing        |
| Embedding      | Not normalized      | L2 normalized                  |
| Retrieval      | Not implemented     | Fully supported                |

---

## 🛠️ Installation

```bash
pip install timm pytorch-metric-learning
```

---

## ▶️ Training

```bash
python train.py
```

---

## 🔎 Inference (Retrieval)

* Extract embeddings for:

  * Query set
  * Gallery set
* Compute cosine similarity
* Rank results

---

## 📁 Project Structure

```
├── train.py
├── model.py
├── dataset.py
├── inference.py
├── README.md
```

---

## 🧪 Future Work

* Cross-attention with spatial tokens
* Transformer-based fusion
* Re-ranking for improved mAP
* Temporal modeling (sequence-based Re-ID)

---

## 👤 Author

Aditya Gupta
B.Tech Engineering Student
Interested in Computer Vision & Deep Learning

---

## ⭐ Acknowledgements

* ICPR TVRID Challenge
* PyTorch Metric Learning Library
* TIMM (EfficientNet models)
