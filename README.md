# Uncertainty-Aware Music Genre Classification using Evidential Deep Learning

This repository contains code and instructions to reproduce the results of our study on uncertainty-aware music genre classification using Evidential Deep Learning (EDL). The model is trained on the GTZAN dataset and evaluated with various uncertainty and calibration metrics.

---

## 📁 Dataset

- **Name**: GTZAN Genre Collection  
- **Source**: [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Description**: 1,000 audio files (30 seconds each), across 10 genres:
  `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

---

## ⚙️ Requirements

Install required packages (for Colab or local setup):

```bash
pip install librosa tensorflow numpy pandas scikit-learn matplotlib tqdm
