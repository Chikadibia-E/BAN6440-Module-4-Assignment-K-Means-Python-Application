"""
kmeans_gsod_2025.py
K-Means clustering pipeline for GSOD 2025 daily summaries.

Dependencies:
  pandas, numpy, scikit-learn, matplotlib, seaborn

Install:
  pip install pandas numpy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Configuration
# -------------------------
CSV_PATH = "gsod_2025.csv"
RANDOM_STATE = 42
MAX_K = 10  # for elbow/silhouette search

# -------------------------
# Utility functions
# -------------------------
def load_gsod(csv_path: str) -> pd.DataFrame:
    """Load GSOD CSV into a DataFrame with basic validation."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded CSV is empty")
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select numeric meteorological features commonly present in GSOD.
    If column names differ, attempt common alternatives.
    """
    # Candidate column names and fallbacks
    candidates = {
        "mean_temp": ["TEMP", "mean_temp", "MEAN_TEMP", "TEMP_C", "TEMP_F"],
        "dew_point": ["DEWP", "dew_point", "DEW_POINT"],
        "sea_level_pressure": ["SLP", "sea_level_pressure", "SEA_LEVEL_PRESSURE"],
        "wind_speed": ["WDSP", "wind_speed", "WIND_SPEED"],
        "max_temp": ["MAX", "max_temp", "MAX_TEMP"],
        "min_temp": ["MIN", "min_temp", "MIN_TEMP"],
        "precipitation": ["PRCP", "precipitation", "PRECIP"]
    }

    selected = {}
    for key, names in candidates.items():
        for n in names:
            if n in df.columns:
                selected[key] = df[n]
                break
    # Build DataFrame from selected features
    feat_df = pd.DataFrame(selected)
    if feat_df.shape[1] < 3:
        raise KeyError("Not enough numeric features found for clustering. Found: " + ", ".join(feat_df.columns))
    return feat_df

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Impute simple missing values and scale features."""
    # Simple imputation: median
    X_imputed = X.fillna(X.median())
    # Remove rows still containing NaN (if any)
    X_imputed = X_imputed.dropna()
    return X_imputed

# -------------------------
# Clustering pipeline
# -------------------------
def build_pipeline(n_components_pca: int = 2, n_clusters: int = 3):
    """Return a scikit-learn pipeline: scaler -> PCA -> KMeans."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components_pca, random_state=RANDOM_STATE)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10))
    ])
    return pipeline

def choose_k(X: np.ndarray, k_max: int = MAX_K):
    """Compute elbow (inertia) and silhouette scores to recommend k."""
    inertias = []
    silhouettes = []
    ks = list(range(2, min(k_max, X.shape[0]-1) + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        inertias.append(km.inertia_)
        labels = km.labels_
        # silhouette requires at least 2 clusters and less than n_samples clusters
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = np.nan
        silhouettes.append(s)
    return ks, inertias, silhouettes

# -------------------------
# Main execution
# -------------------------
def main():
    # Load
    df = load_gsod(CSV_PATH)

    # Select features (based on GSOD elements: mean temp, dew point, pressure, wind, max/min, precipitation).
    X_raw = select_features(df)

    # Preprocess
    X_clean = preprocess_features(X_raw)

    # Optionally reduce dimensionality for visualization
    pca_for_k = PCA(n_components=min(5, X_clean.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca_for_k.fit_transform(StandardScaler().fit_transform(X_clean))

    # Choose k using elbow and silhouette
    ks, inertias, silhouettes = choose_k(X_pca, k_max=MAX_K)

    # Plot elbow and silhouette (save figures)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(ks, inertias, marker='o')
    plt.title("Elbow: inertia vs k")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.subplot(1,2,2)
    plt.plot(ks, silhouettes, marker='o')
    plt.title("Silhouette vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig("k_selection.png", dpi=150)

    # Choose k (example: pick k with max silhouette if available)
    if not all(np.isnan(silhouettes)):
        best_k = ks[int(np.nanargmax(silhouettes))]
    else:
        # fallback: elbow heuristic (first large drop)
        diffs = np.diff(inertias)
        best_k = ks[int(np.argmin(diffs))] if len(diffs) > 0 else 3

    # Build final pipeline with PCA to 2 components for visualization
    pipeline = build_pipeline(n_components_pca=2, n_clusters=best_k)
    pipeline.fit(X_clean)

    # Attach cluster labels to original DataFrame (align by index)
    labels = pipeline.named_steps["kmeans"].labels_
    result = df.loc[X_clean.index].copy()
    result["cluster"] = labels

    # Save cluster assignments and cluster centers (in original feature space approximate via inverse transform)
    result.to_csv("gsod_2025_clusters.csv", index=False)
    # Save cluster centers in PCA space and approximate original space
    centers_pca = pipeline.named_steps["kmeans"].cluster_centers_
    np.save("cluster_centers_pca.npy", centers_pca)

    print(f"Completed clustering with k={best_k}. Results saved to gsod_2025_clusters.csv and figures saved.")

if __name__ == "__main__":
    main()
