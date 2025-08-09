# 🐱 Real-Time Object Detection & Robot Control

This project implements **real-time object detection** and **autonomous robot control** using a **Raspberry Pi 5** and **deep learning**.  
Our main goal: Enable a Fischertechnik robot to detect and chase a specific object – in this case, a **cat** – using computer vision and control engineering.

---

## 📌 Overview

- **Object Detection Model:** YOLOv8n (fine-tuned for single-class cat detection)
- **Hardware:** Raspberry Pi 5, webcam, Fischertechnik robot
- **Inference Engine:** NCNN (Tencent) – optimized for ARM-based edge devices
- **Control System:**  
  - Lateral motion – **PD Controller** (keeps object centered in camera frame)  
  - Longitudinal motion – **P Controller** (maintains desired distance)
- **Dataset:** Custom **mega-dataset of cats**, balanced with non-cat images to avoid false positives.

---

## 🗂 Dataset Generation

We created a **17,588-image** dataset by combining:
- COCO Dataset
- Roboflow Cats Dataset
- Custom collected cat images

**Key steps:**
1. **Balancing Classes** – Equal ratio of cat vs. non-cat images.
2. **Diversity** – Multiple cat breeds and backgrounds.
3. **Augmentation Techniques** – Flips, rotations, brightness/contrast, HSV adjustments.
4. **Annotation** – YOLOv8n-assisted labeling.

---

## 🧠 Model Selection & Training

We evaluated YOLOv5, YOLOv8n, SSD, Faster R-CNN, EfficientDet, and RetinaNet.  
**YOLOv8n** was chosen for:
- High inference speed on Raspberry Pi
- Small model size
- Easy export to NCNN format

**Training strategy:**
- Freeze Backbone & Neck (first 21 layers)
- Fine-tune Head (last layer) for cat classification
- Two-stage training: 30 epochs + 20 epochs

---

## ⚡ Deployment on Raspberry Pi 5

We benchmarked multiple formats for inference speed:

| Format      | Inference Time (ms) | FPS   |
|-------------|--------------------|-------|
| TorchScript | 456                | 1.5–2 |
| PyTorch (.pt)| 350                | 2.5–3 |
| ONNX        | 300                | 3–3.5 |
| **NCNN**    | **110**            | **8–9** |

✅ **NCNN** was selected for **real-time performance**.

---

## 🎯 Control Engineering

### Lateral Motion (PD Controller)
- Keeps the detected object centered in the camera frame.
- Error = horizontal offset between bounding box center and frame center.

### Longitudinal Motion (P Controller)
- Maintains optimal distance using bounding box area.
- Target area = 80% of frame area.

---

## 🔧 Controller Tuning Insights
- **PD Controller:** Low-to-moderate derivative gain offers a good trade-off between responsiveness and stability.
- **P Controller:** Stable convergence but slower response; chosen for simplicity and hardware limitations.

---

## 📸 Project Demo
*(Add images or GIFs of the robot in action here)*

---

## 📚 References
1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
2. [Microsoft COCO Dataset](https://cocodataset.org/)
3. [Roboflow Cats Dataset](https://universe.roboflow.com/mohamed-traore-2ekkp/cats-n9b87)

---

## 🏆 Key Takeaways
- **End-to-end system**: dataset preparation → model fine-tuning → hardware deployment → control loop tuning.
- **Edge AI focus**: Optimized for Raspberry Pi with NCNN.
- **Practical robotics application**: Combining computer vision and control engineering for autonomous chasing behavior.

---

## 📜 License
This project is released under the MIT License – feel free to use, modify, and share.

---

## 🤝 Contributors
- **Shantanu Shirsath**  
- **Aakash Deshpande**  

University of Siegen

---
