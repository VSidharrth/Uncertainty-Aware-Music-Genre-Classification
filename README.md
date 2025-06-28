
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
```

- Python ≥ 3.8  
- TensorFlow ≥ 2.18  
- Libraries: `librosa`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`

---

## 🧠 Methodology Overview

### Feature Extraction:
- Extract 40 MFCCs per 3-second segment with 50% overlap from each 30s audio file.  
- Each segment is represented as a 2D input with shape: `(130, 40, 1)`

### Model Architecture:
- CNN + LSTM model  
- Output activation: **Softplus** to generate Dirichlet parameters  
- Loss function: Negative Log Likelihood + KL Divergence (Evidential Deep Learning)

### Uncertainty Estimation:
- Based on Dirichlet distribution: `alpha = evidence + 1`  
- Predictive uncertainty = `num_classes / sum(alpha)`

### Calibration:
- **Temperature scaling** is applied to the predicted Dirichlet probabilities to reduce Expected Calibration Error (ECE)

### Evaluation Metrics:
- Accuracy  
- Macro F1-score  
- Confusion Matrix  
- Expected Calibration Error (ECE)  
- Reliability Diagram  
- Selective Prediction (Accuracy vs Coverage)

---

## ▶️ How to Run

### 1. Mount Google Drive (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Extract MFCC features and segment the audio.

### 3. Train the CNN-LSTM EDL model:
```python
model.fit(X_train, y_train, ...)
```

### 4. Predict genre labels and compute uncertainty scores.

### 5. Plot:
- Confusion matrix  
- Selective prediction curves  
- Reliability diagrams

### 6. Apply temperature scaling:
```python
temperature_scale(probs, optimal_temp)
```

---

## 📊 Results Summary

| Metric                        | Value   |
|------------------------------|---------|
| Test Accuracy                | 79.72%  |
| Macro F1-score               | ~0.80   |
| ECE (before calibration)     | 0.2918  |
| ECE (after calibration)      | 0.0190  |
| Accuracy (Top 50% confident) | ~94%    |

---

## 📌 Citation

If you use this code or results in your research, please cite:

> V. Sidharrth, J. Sarada, B. Alatas. *Uncertainty-Aware Music Genre Classification using Evidential Deep Learning*. PeerJ Computer Science (Under Review, 2025).

---

## 📝 License

For academic research and educational use only. Please contact the authors for any other usage.
