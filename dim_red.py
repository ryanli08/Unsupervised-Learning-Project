import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn import metrics
from collections import defaultdict
from itertools import product
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.metrics import mean_squared_error

def explained_variance_plot(pca, filename):
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    max_components = min(80,np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else len(pca.explained_variance_))
    components = range(1, max_components + 1)

    expl_var = pca.explained_variance_ratio_[:max_components]
    cum_var_trimmed = cum_var[:max_components]

    fig, ax1 = plt.subplots()
    ax1.plot(components, expl_var, marker='o', label='Explained Variance')
    ax1.plot(components, cum_var_trimmed, marker='x', linestyle='--', label='Cumulative Variance')

    if np.any(cum_var >= 0.95):
        cutoff_idx = np.argmax(cum_var >= 0.95) + 1
        ax1.axvline(x=cutoff_idx, color='red', linestyle=':', label=f'95% variance at {cutoff_idx} comps')

    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Variance Ratio')
    ax1.set_title(f'{filename} - PCA Explained Variance')
    ax1.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Images/{filename}_pca_variance.png")
    plt.close()

def reconstruction_error_plot(X, filename):
    from sklearn.decomposition import FastICA

    transformer_map = {
        "PCA": PCA,
        "FastICA": lambda n, seed: FastICA(
            n_components=n, random_state=seed, whiten='unit-variance', max_iter=2000
        ),
        "RP": lambda n, seed: GRP(n_components=n, random_state=seed)
    }

    line_styles = {
        "PCA": {"linestyle": "-", "color": "blue", "linewidth": 4},
        "FastICA": {"linestyle": "-", "color": "orange", "linewidth": 1.5},
        "RP": {"linestyle": "-", "color": "green", "linewidth": 1.5}
    }

    n_features = X.shape[1]
    max_components = min(n_features, 20)
    n_components = range(1, max_components + 1)
    n_trials = 10
    plt.figure(figsize=(10, 6))

    for name, transformer_func in transformer_map.items():
        errors = defaultdict(list)
        for n in n_components:
            for i in range(n_trials):
                try:
                    if name == "PCA":
                        transformer = PCA(n_components=n)
                    else:
                        transformer = transformer_func(n, i)

                    X_reduced = transformer.fit_transform(X)

                    if hasattr(transformer, 'inverse_transform'):
                        X_recon = transformer.inverse_transform(X_reduced)
                    else:
                        inv = np.linalg.pinv(transformer.components_.T)
                        X_recon = X_reduced @ inv

                    mse = metrics.mean_squared_error(X, X_recon)
                except Exception:
                    mse = np.nan
                errors[n].append(mse)

        mean_err = np.array([np.nanmean(errors[n]) for n in n_components])
        std_err = np.array([np.nanstd(errors[n]) for n in n_components])

        style = line_styles.get(name, {"linestyle": "-", "color": None, "linewidth": 1.5})
        plt.plot(n_components, mean_err,
                 linestyle=style["linestyle"],
                 color=style["color"],
                 linewidth=style["linewidth"],
                 label=f"{name} Mean Error")

        plt.fill_between(n_components, mean_err - std_err, mean_err + std_err,
                         alpha=0.2, color=style["color"])

    plt.title(f"{filename} - Combined Reconstruction Error (PCA, FastICA, RP)")
    plt.xlabel("Number of Components")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Images/{filename}_combined_recon_error.png")
    plt.close()




def kurtosis_plot(X, filename):
    n_features = X.shape[1]
    max_components = min(n_features, 53)
    kurt_vals = []

    for n in range(1, max_components + 1):
        ica = FastICA(
            n_components=n,
            random_state=0,
            max_iter=2000,
            tol=1e-3,
            algorithm='deflation',
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                X_ica = ica.fit_transform(X)
                kurtosis = pd.DataFrame(X_ica).kurt(axis=0).abs().mean()
            except Exception:
                kurtosis = np.nan

        kurt_vals.append(kurtosis)

    optimal_k = np.argmax(kurt_vals) + 1

    plt.figure()
    plt.plot(range(1, max_components + 1), kurt_vals, marker='o')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'{filename} - ICA Average Kurtosis')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Kurtosis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Images/{filename}_ica_kurtosis.png")
    plt.close()

    return optimal_k

def find_optimal_rp_components(X, filename):

    n_features = X.shape[1]
    n_trials = 10
    max_components = n_features
    best_n = None
    best_err = float('inf')

    print(f"\n{filename.upper()} - RP Component Search (up to {max_components})")

    for n in range(1, max_components + 1):
        errs = []
        for seed in range(n_trials):
            try:
                rp = GRP(n_components=n, random_state=seed)
                X_reduced = rp.fit_transform(X)

                # Reconstruct using pseudo-inverse
                inv = np.linalg.pinv(rp.components_.T)
                X_recon = X_reduced @ inv

                mse = mean_squared_error(X, X_recon)
                errs.append(mse)
            except Exception:
                continue

        mean_err = np.mean(errs) if errs else np.inf
        print(f"  Components: {n:2d} | MSE: {mean_err:.6f}")

        if mean_err < best_err:
            best_err = mean_err
            best_n = n

    print(f"=> Best RP components for {filename}: {best_n} with MSE: {best_err:.6f}\n")
    return best_n


def analyze_dimensionality_reduction(X, filename):
    print(f"Analyzing dimensionality reduction for: {filename}")
    pca = PCA(random_state=0)
    pca.fit(X)
    explained_variance_plot(pca, filename)
    reconstruction_error_plot(X, filename)
    kurtosis_plot(X, filename)
    best_rp = find_optimal_rp_components(X, filename)




if __name__ == "__main__":
    # Bankruptcy dataset
    df = pd.read_csv("data/bankruptcy_data.csv")
    X = df.drop(columns=["Bankrupt?"], errors="ignore").select_dtypes(include='number')
    X_std = StandardScaler().fit_transform(X)
    analyze_dimensionality_reduction(X_std, "bankruptcy")

    # Cancer dataset
    df1 = pd.read_csv("data/global_cancer_patients_2015_2024.csv")
    cols_to_keep = ["Age", "Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
    X1 = df1[cols_to_keep]
    X_std1 = StandardScaler().fit_transform(X1)
    analyze_dimensionality_reduction(X_std1, "cancer")



