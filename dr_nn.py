import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess
df = pd.read_csv("data/bankruptcy_data.csv").dropna()
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Shared train/test split
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=1387, stratify=y
)

# Fit DR methods once
pca = PCA(n_components=53, random_state=42).fit(X_train_full)
ica = FastICA(n_components=8, random_state=42).fit(X_train_full)
rp = GRP(n_components=30, random_state=42).fit(X_train_full)

dr_methods = {
    "Full": (X_train_full, X_test_full),
    "RP": (rp.transform(X_train_full), rp.transform(X_test_full)),
    "PCA": (pca.transform(X_train_full), pca.transform(X_test_full)),
    "ICA": (ica.transform(X_train_full), ica.transform(X_test_full)),
}

# NN config
nn_config = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "max_iter": 500,
    "random_state": 1387
}

results = []

# Evaluate each
for method, (X_train, X_test) in dr_methods.items():
    clf = MLPClassifier(**nn_config)
    print(f"Training on {method} features...")
    start = time.time()
    clf.fit(X_train, y_train)
    duration = time.time() - start
    acc = accuracy_score(y_test, clf.predict(X_test))
    results.append((method, acc, duration))
    print(f"Accuracy: {acc:.4f}, Training Time: {duration:.2f}s\n")

# Accuracy plot
methods, accuracies, times = zip(*results)
plt.figure(figsize=(8, 5))
plt.bar(methods, accuracies, color='mediumseagreen')
plt.title("NN Accuracy by Dimensionality Reduction - Bankruptcy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("images/nn_dr_accuracy_bankruptcy.png")
plt.show()

# Training time plot
plt.figure(figsize=(8, 5))
plt.bar(methods, times, color='orange')
plt.title("NN Training Time by Dimensionality Reduction - Bankruptcy")
plt.ylabel("Training Time (s)")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("images/nn_dr_training_time_bankruptcy.png")
plt.show()
