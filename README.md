# Unsupervised Learning Project
### Clustering and Dimensionality Reduction on Cancer and Bankruptcy Data  
**Author:** Ryan Li | M.S. Computer Science (Machine Learning), Georgia Tech  

---

## 🧠 Overview  
This project applies **unsupervised learning techniques**—clustering and dimensionality reduction—on two real-world datasets:  
- **Bankruptcy Data** (95 financial indicators)  
- **Global Cancer Patients Data (2015–2024)**  

The study explores how **K-Means** and **Expectation Maximization (EM)** uncover structure in high- and low-dimensional data, both before and after dimensionality reduction using:  
- **Principal Component Analysis (PCA)**  
- **Independent Component Analysis (ICA)**  
- **Random Projection (RP)**  

Additionally, a **neural network classifier** was retrained on the reduced feature spaces to assess downstream effects on accuracy and training time.

---

## ⚙️ Methods
- Clustering evaluated across \(k \in \{2, \ldots, 9\}\) using metrics:  
  - Silhouette Score  
  - Adjusted Rand Index (ARI)  
  - Calinski-Harabasz Index (CHI)  
  - Bayesian Information Criterion (BIC)  

- Dimensionality reduction was performed using:  
  - **PCA** (95% variance retained)  
  - **ICA** (components maximizing kurtosis)  
  - **RP** (minimizing reconstruction error)  

- A fixed neural network (ReLU, layers [64, 32]) compared model performance on full vs. reduced data.

---

## 📊 Key Results  
| Method | Best Metric | Observation |
|--------|--------------|-------------|
| **K-Means (Bankruptcy)** | Silhouette ≈ 0.13 | Clusters well-separated geometrically but poor label alignment. |
| **EM (Bankruptcy)** | Silhouette ≈ 0.41 | Best geometric clusters; weak ARI → clusters not matching labels. |
| **PCA vs ICA** | Identical recon. error | Data likely Gaussian — ICA converges to PCA-like components. |
| **Cancer Dataset** | All DR methods similar | Low feature count (6) limits clustering improvement. |
| **Neural Network (Bankruptcy)** | PCA: 0.9614 acc. | Fastest training (29s) with minimal accuracy loss. |

---

## 🧩 Insights  
- **EM** outperformed K-Means in capturing geometric cluster structure for high-dimensional data.  
- **Dimensionality reduction** improved cluster compactness and efficiency, especially PCA and ICA.  
- **Random Projection**, while efficient, introduced distortion—stronger for high-dimensional data.  
- **Neural network retraining** confirmed that PCA offers the best balance between speed and performance.  

---

## 🗂️ Project Structure  

