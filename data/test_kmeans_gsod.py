# test_kmeans_gsod.py
import pandas as pd
import numpy as np
import pytest
from kmeans_gsod_2025 import load_gsod, select_features, preprocess_features, build_pipeline

CSV_PATH = "C:/Users/e_chi/OneDrive - Nexford University/BAN6440 - Assignments/Module 4/data/gsod_2025.csv"

def test_load_nonempty():
    df = load_gsod(CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_select_features_returns_dataframe():
    df = load_gsod(CSV_PATH)
    feats = select_features(df)
    assert isinstance(feats, pd.DataFrame)
    assert feats.shape[1] >= 3  # at least 3 numeric features

def test_preprocess_no_nan_after_impute():
    df = load_gsod(CSV_PATH)
    feats = select_features(df)
    clean = preprocess_features(feats)
    assert clean.isna().sum().sum() == 0

def test_pipeline_runs():
    df = load_gsod(CSV_PATH)
    feats = select_features(df)
    clean = preprocess_features(feats)
    pipeline = build_pipeline(n_components_pca=2, n_clusters=3)
    pipeline.fit(clean)
    labels = pipeline.named_steps["kmeans"].labels_
    assert len(labels) == clean.shape[0]
