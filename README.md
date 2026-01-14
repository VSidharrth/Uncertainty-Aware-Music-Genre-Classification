# Uncertainty-Aware Music Genre Classification using Evidential Deep Learning

This repository contains the complete code and experimental setup to reproduce the results presented in our study on **uncertainty-aware music genre classification** using **Evidential Deep Learning (EDL)**. The proposed CNN–LSTM-based evidential framework is evaluated on the **GTZAN dataset**, with a focus on **uncertainty quantification, calibration, and reliability-aware evaluation**.

---

## 📁 Dataset

- **Name**: GTZAN Genre Collection  
- **Source**: GTZAN Dataset on Kaggle (MARSYAS collection)  
- **Description**:
  - 1,000 audio recordings  
  - Duration: 30 seconds per file  
  - Sampling rate: 22050 Hz  
  - Mono channel, 16-bit  
  - 10 genres:
    ```
    blues, classical, country, disco, hiphop,
    jazz, metal, pop, reggae, rock
    ```

⚠️ One corrupted Jazz audio file is automatically discarded during preprocessing.

## ⚙️ Requirements

### Software
- Python **3.11.13**
- TensorFlow **2.18.0**

### Required Libraries
```bash
pip install librosa tensorflow numpy pandas scikit-learn matplotlib tqdm
```
### Libraries Used

- `librosa`
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

## 🔹 Dataset Splitting (Song-Level)

- Segments from a single song are restricted to **one split only**
- **Training set**: 64%  
  - 5% removed using **Local Outlier Factor (LOF)**
- **Validation set**: 16%
- **Test set**: 20%

### Dataset Statistics

| Description | Samples |
|------------|---------|
| Training samples | 12,137 |
| Validation samples | 3,037 |
| Testing samples | 3,798 |
| Total samples (before outlier removal) | 18,972 |
| Total samples (after outlier removal) | 18,365 |

---

## 🔹 Outlier Detection

- Algorithm: **Local Outlier Factor (LOF)**
- Applied only to **training data**
- Number of neighbors: **20**
- Contamination factor: **0.05**
- Outliers removed: **607**

## 🧱 Model Architecture

A **CNN–LSTM Evidential Neural Network** is employed.

| Layer | Output Shape | Parameters |
|------|--------------|------------|
| Input | (None, 130, 40, 1) | 0 |
| Conv2D (32, 3×3, ReLU) | (None, 130, 40, 32) | 320 |
| MaxPooling2D (2×2) | (None, 65, 20, 32) | 0 |
| Reshape | (None, 65, 640) | 0 |
| LSTM (64 units) | (None, 64) | 180,480 |
| Dropout | (None, 64) | 0 |
| Dense (Softplus, 10) | (None, 10) | 650 |

**Total parameters**: 181,450  
**Trainable parameters**: 181,450  
**Non-trainable parameters**: 0

## 🧮 Evidential Deep Learning Framework

- Output activation: **Softplus**
- Evidence computation:
  
  \[
  e_k = \text{softplus}(z_k)
  \]

- Dirichlet parameters:
  
  \[
  \alpha_k = e_k + 1
  \]

- Dirichlet concentration:
  
  \[
  S = \sum_{k=1}^{K} \alpha_k
  \]

- Predictive probability:
  
  \[
  \hat{p}_k = \frac{\alpha_k}{S}
  \]

---

## 🔻 Loss Function

The total training loss is defined as:

\[
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda \cdot \mathcal{L}_{\text{KL}}
\]

where:
- \(\lambda = 1\) is the regularization coefficient
- \(\mathcal{L}_{\text{NLL}}\): **Negative Log-Likelihood** loss
- \(\mathcal{L}_{\text{KL}}\): **Kullback–Leibler divergence** between the predicted Dirichlet distribution and a uniform prior

## 🏋️ Training Configuration

- Optimizer: **Adam**
- Learning rate: **1e−3**
- Batch size: **32**
- Epochs: **40**
- Metric: **Categorical accuracy**

---

## 🔍 Uncertainty Quantification

Uncertainty is computed as:

\[
u = \frac{K}{S}
\]

Where:
- \(K = 10\) (number of genres)
- \(S\) = Dirichlet concentration

### Reliability Threshold

- **Reliable prediction**: \(u < 0.4\)
- **Unreliable prediction**: \(u \geq 0.4\)

## 📊 Evaluation Metrics

- Classification accuracy
- Macro F1-score
- Confusion matrix
- Expected Calibration Error (ECE)
- Reliability diagram
- Selective prediction (Accuracy vs. Coverage)

---

## 📈 Results Summary

| Metric | Value |
|------|------|
| Training Accuracy | 80.64% |
| Validation Accuracy | 72.99% |
| Test Accuracy | 65.81% with song level splitting, 76.06% without song level splitting|
| Test Loss | 1.2574 |
| ECE (before calibration) | 0.1401 |
| ECE (after calibration) | 0.0791 |
| Optimal temperature | 0.6399 |

---

### 🎯 Selective Prediction

- High-confidence samples achieve near-perfect accuracy
- Accuracy improves as coverage decreases
- Temperature scaling significantly enhances prediction reliability

---

## 📐 Statistical Validation

- Statistical test: **Wilcoxon signed-rank test**
- Bootstrap iterations: **1000**
- Mean ECE reduction: **0.060845**
- 95% confidence interval: **[0.034680, 0.087250]**
- Test statistic: **0**
- p-value: **3.325859 × 10⁻¹⁶⁵**

## ⚠️ Limitations

- Only MFCC features are used
- GTZAN dataset contains inherent genre ambiguity
- Evaluation is performed in an offline setting

---

## 📌 Citation

If you use this work, please cite:

> V. Sidharrth, J. Sarada, B. Alatas  
> **Uncertainty-Aware Music Genre Classification using Evidential Deep Learning**  
> *PeerJ Computer Science* (Under Review, 2025)

---

## 📝 License & Contact
This work is intended strictly for **academic and educational purposes**.

**V. Sidharrth**  
Email: bl.en.u4aid23054@bl.students.amrita.edu  

**J. Sarada**  
Email: j_sarada@blr.amrita.edu  

**B. Alatas**  
Email: balatas@firat.edu.tr
