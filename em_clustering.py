import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

def bench_em(estimator, labels, X):
    estimator.fit(X)
    predictions = estimator.predict(X)
    try:
        sil_score = metrics.silhouette_score(X, predictions)
    except ValueError:
        sil_score = 0.0

    return {
        'v_measure': metrics.v_measure_score(labels, predictions),
        'ari': metrics.adjusted_rand_score(labels, predictions),
        'silhouette': sil_score,
        'dbi': metrics.davies_bouldin_score(X, predictions),
        'chi': metrics.calinski_harabasz_score(X, predictions),
        'bic': estimator.bic(X)
    }



def plot_combined_scores(n_clusters, scores, filename):
    silhouette_scores = [s['silhouette'] for s in scores]
    ari_scores = [s['ari'] for s in scores]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score', color='tab:blue')
    ax1.plot(n_clusters, silhouette_scores, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Adjusted Rand Index', color='tab:red')
    ax2.plot(n_clusters, ari_scores, marker='x', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f"EM Scores: {filename}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"Images/{filename}_combined_scores.png")
    plt.close()

def get_best_k_table(n_clusters, scores_list, dataset_name):
    df = pd.DataFrame(scores_list, index=n_clusters)
    best_sil_k = df['silhouette'].idxmax()
    best_ari_k = df['ari'].idxmax()

    result_df = pd.DataFrame({
        "metric": ["silhouette", "ari"],
        "best_k": [best_sil_k, best_ari_k],
        "silhouette": [df.loc[best_sil_k]['silhouette'], df.loc[best_ari_k]['silhouette']],
        "ari": [df.loc[best_sil_k]['ari'], df.loc[best_ari_k]['ari']],
        "v_measure": [df.loc[best_sil_k]['v_measure'], df.loc[best_ari_k]['v_measure']],
        "dbi": [df.loc[best_sil_k]['dbi'], df.loc[best_ari_k]['dbi']],
        "chi": [df.loc[best_sil_k]['chi'], df.loc[best_ari_k]['chi']],
        "bic": [df.loc[best_sil_k]['bic'], df.loc[best_ari_k]['bic']]
    })

    result_df = result_df.round(2)

    # generate table
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    table = ax.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    plt.savefig(f"Images/{dataset_name}_em_summary_table.png")

    plt.close()

    return result_df

def run_em(file_path, dataset_name):
    df = pd.read_csv(file_path)

    if dataset_name == "cancer":
        # Bin severity into 3 levels
        df["Severity_Binned"] = pd.cut(
            df["Target_Severity_Score"],
            bins=[0, 3.5, 6.5, np.inf],
            labels=[0, 1, 2]
        )
        label_col = "Severity_Binned"
    elif dataset_name == "bankruptcy":
        label_col = "Bankrupt?"
    else:
        raise ValueError("Unknown dataset_name. Expected 'cancer' or 'bankruptcy'.")

    # Extract features and labels
    y = df[label_col].astype(int).values
    X = df.select_dtypes(include="number").drop(columns=[label_col], errors="ignore").values
    X_scaled = StandardScaler().fit_transform(X)

    n_clusters = list(range(2, 10))
    scores = []
    for k in n_clusters:
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=1, max_iter=100, random_state=42)
        scores.append(bench_em(gmm, y, X_scaled))

    plot_combined_scores(n_clusters, scores, f"em_{dataset_name}")
    summary_table = get_best_k_table(n_clusters, scores, f"em_{dataset_name}")
    print(f"\n=== Summary for {dataset_name} ===")
    print(summary_table)
    return summary_table

if __name__ == "__main__":
    run_em("data/bankruptcy_data.csv", "bankruptcy")
    run_em("data/global_cancer_patients_2015_2024.csv", "cancer")
