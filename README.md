# 🧠 Comparative Analysis of Generative Models for Image Synthesis and Anomaly Detection

This repository contains my implementation and analysis of three foundational **Generative AI models** — **GAN**, **CGAN**, and **VAE** — as part of the *Generative AI* course at **FAST–NUCES Lahore**.

---

## 📘 Overview

The project performs a **comparative evaluation** of three generative architectures for:
- **Image Synthesis** on MNIST and CIFAR-10 datasets.
- **Anomaly Detection** using a Variational Autoencoder (VAE) on a real-world Credit Card Fraud dataset.

---

## ⚙️ Models Implemented

### 🔹 Generative Adversarial Network (GAN)
- Implemented with DCGAN-style convolutional layers.
- Trained on MNIST and CIFAR-10 datasets.
- Evaluated using **Fréchet Inception Distance (FID)** and **Kernel Inception Distance (KID)**.

### 🔹 Conditional GAN (CGAN)
- Added label conditioning to control generated image classes.
- Demonstrated effective results on MNIST but unstable behavior on CIFAR-10.

### 🔹 Variational Autoencoder (VAE)
- Implemented with encoder–decoder architecture.
- Used for both image generation and anomaly detection.
- Achieved **93% accuracy** and **0.93 F1-score** on fraud detection.

---

## 🧩 Methodology

- **Datasets Used:**
  - MNIST – Handwritten digits
  - CIFAR-10 – Natural color images
  - Credit Card Fraud Dataset – Highly imbalanced tabular data

- **Evaluation Metrics:**
  - *FID* and *KID* for generative quality
  - *Precision*, *Recall*, and *F1-score* for anomaly detection

---

## 📊 Key Results

| Model | Dataset | FID ↓ | KID ↓ | Notes |
|--------|----------|--------|--------|-------|
| GAN | MNIST | 24.26 | 0.0123 | Best visual quality |
| CGAN | MNIST | 198.61 | 0.254 | Training instability |
| VAE | MNIST | 88.35 | 0.0708 | Smooth latent clustering |
| VAE (Fraud Detection) | Credit Card Data | – | – | **Accuracy: 93%, F1: 0.93** |

---

## 🔬 Insights
- **GANs** excel in realism and quality.
- **CGANs** offer control but require stability tuning.
- **VAEs** provide meaningful latent spaces, ideal for anomaly detection tasks.

---

## 📂 Repository Contents
- `GenAI_Assignment_01.ipynb` – Main implementation notebook  
- `GenAI_Assignment_Report.pdf` – Full technical report with results and discussion  

---

## 🧑‍💻 Author
**Muhammad Nouman Hanif**  
MS (Data Science) Candidate | FAST–NUCES Lahore  
📧 Email: [mnoumanhanif66@gmail.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/mnoumanhanif/)  
