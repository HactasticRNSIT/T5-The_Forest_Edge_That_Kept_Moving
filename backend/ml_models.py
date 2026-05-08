"""
EcoShift AI — Module 3: ML Models
Random Forest + SHAP + PCA + Granger + K-Means

Usage: python3 backend/ml_models.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap

warnings.filterwarnings("ignore")

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "ndvi_2018", "ndvi_2021", "ndvi_2024",
    "ndvi_change_2015_2018", "ndvi_change_2018_2021",
    "ndvi_change_2021_2024", "ndvi_total_change",
    "ndvi_mean", "ndvi_std", "ndvi_accel",
    "fragmentation_index",
    "road_length_km", "building_count",
    "infra_pressure", "urban_proximity_score",
    "dist_to_cbd_km",
    "rainfall_anomaly", "rainfall_stress",
    "gbif_observations", "gbif_species_count",
    "biodiversity_density",
    "lag_stress_score",
]


def load_data():
    print("\n[1/5] Loading feature matrix...")
    df = pd.read_csv(os.path.join(PROC_DIR, "ml_features.csv"))

    needs_labels = (
        "degradation_label" not in df.columns or
        df["degradation_label"].isna().all() or
        (df["degradation_label"].astype(str) == "nan").all()
    )

    if needs_labels:
        print("    [WARN] Labels missing — regenerating from EVS...")
        if "ecological_vulnerability_score" not in df.columns:
            ndvi_loss   = df["ndvi_total_change"].fillna(0).clip(-1, 0).abs()
            ndvi_loss_n = ndvi_loss / (ndvi_loss.max() + 1e-6)
            frag_n      = df["fragmentation_index"].fillna(0) / (df["fragmentation_index"].fillna(0).max() + 1e-6)
            infra_n     = df["infra_pressure"].fillna(0)
            urb_n       = df["urban_proximity_score"].fillna(0) / (df["urban_proximity_score"].fillna(0).max() + 1e-6)
            df["ecological_vulnerability_score"] = (
                0.35 * ndvi_loss_n + 0.25 * frag_n +
                0.25 * infra_n    + 0.15 * urb_n
            ).round(4)

        evs = df["ecological_vulnerability_score"].fillna(0)
        df["degradation_label"] = pd.cut(
            evs,
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["stable", "at_risk", "degraded"]
        ).astype(str)

    df = df[df["degradation_label"].notna()]
    df = df[~df["degradation_label"].isin(["nan", "", "None"])]
    df = df.reset_index(drop=True)

    print(f"    ✓ {len(df)} rows loaded")
    print(f"    Labels:\n{df['degradation_label'].value_counts()}")
    return df


def train_model(df: pd.DataFrame):
    print("\n[2/5] Training Random Forest classifier...")

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["degradation_label"]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"    ✓ Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    with open(os.path.join(MODELS_DIR, "rf_ecoshift.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(available, f)

    df = df.copy().reset_index(drop=True)
    df["predicted_label"] = le.inverse_transform(model.predict(X.fillna(0)))
    df["pred_confidence"] = model.predict_proba(X.fillna(0)).max(axis=1).round(4)

    print(f"    ✓ Model saved → models/rf_ecoshift.pkl")
    return model, le, available, df


def compute_shap(model, df: pd.DataFrame, feature_cols: list):
    print("\n[3/5] Computing SHAP values...")

    X        = df[feature_cols].fillna(0)
    X_sample = X.sample(min(300, len(X)), random_state=42).reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle multiclass safely
    if isinstance(shap_values, list):
        sv = np.mean([np.abs(s) for s in shap_values], axis=0)
    else:
        sv = np.abs(shap_values)

    # Ensure 2D
    if sv.ndim == 3:
        sv = sv.mean(axis=0)

    mean_abs = np.mean(sv, axis=0)

    # Ensure lengths match
    n_feats = min(len(feature_cols), len(mean_abs))
    importance_df = pd.DataFrame({
        "feature":    feature_cols[:n_feats],
        "importance": mean_abs[:n_feats].round(6),
    }).sort_values("importance", ascending=False)

    print("    Top 5 degradation drivers:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']:35s}  {row['importance']:.4f}")

    importance_df.to_csv(os.path.join(PROC_DIR, "feature_importance.csv"), index=False)
    importance_df.to_csv(os.path.join(PROC_DIR, "shap_values.csv"), index=False)

    print(f"    ✓ SHAP saved → data/processed/feature_importance.csv")
    return importance_df, importance_df


def compute_pca(df: pd.DataFrame, feature_cols: list):
    print("\n[4/5] Computing PCA latent stress axes...")

    X    = df[feature_cols].fillna(0)
    pca  = PCA(n_components=3, random_state=42)
    comp = pca.fit_transform(X)

    ev = pca.explained_variance_ratio_
    print(f"    Variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  PC3={ev[2]:.1%}")

    df = df.copy()
    df["latent_stress_1"]     = comp[:, 0].round(4)
    df["latent_stress_2"]     = comp[:, 1].round(4)
    df["latent_stress_3"]     = comp[:, 2].round(4)
    df["latent_stress_score"] = np.sqrt(comp[:, 0]**2 + comp[:, 1]**2).round(4)

    ls_min = df["latent_stress_score"].min()
    ls_max = df["latent_stress_score"].max()
    df["latent_stress_norm"] = (
        (df["latent_stress_score"] - ls_min) / (ls_max - ls_min + 1e-6)
    ).round(4)

    with open(os.path.join(MODELS_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"    ✓ Latent stress scores computed")
    return df, pca


def compute_granger(df: pd.DataFrame):
    print("\n[5/5] Computing Granger causality (statsmodels)...")
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.api import VAR

    ndvi_years = {2018: "ndvi_2018", 2021: "ndvi_2021", 2024: "ndvi_2024"}
    results = []

    zones = df["zone"].unique() if "zone" in df.columns else ["all"]

    for zone in zones:
        zone_df = df[df["zone"] == zone] if "zone" in df.columns else df
        if len(zone_df) < 10:
            continue

        ts_data = []
        for year, col in ndvi_years.items():
            if col in zone_df.columns:
                ts_data.append({
                    "year":      year,
                    "ndvi_mean": zone_df[col].mean(),
                    "infra":     zone_df["infra_pressure"].mean(),
                    "rainfall":  zone_df["rainfall_anomaly"].fillna(0).mean(),
                })

        ts = pd.DataFrame(ts_data).sort_values("year")
        if len(ts) < 3:
            continue

        ndvi  = ts["ndvi_mean"].values
        infra = ts["infra"].values

        # Lag-1 correlation
        lag_corr = float(np.corrcoef(infra[:-1], ndvi[1:])[0, 1]) if len(ndvi) >= 2 else 0.0

        # Try real Granger causality if enough data
        p_value = None
        try:
            data = pd.DataFrame({"ndvi": ndvi, "infra": infra})
            gc_result = grangercausalitytests(data[["ndvi", "infra"]], maxlag=1, verbose=False)
            p_value = round(gc_result[1][0]["ssr_ftest"][1], 4)
        except Exception:
            pass

        results.append({
            "zone":                zone,
            "infra_ndvi_lag_corr": round(lag_corr, 4),
            "granger_p_value":     p_value if p_value is not None else "N/A",
            "interpretation": (
                "Infrastructure significantly predicts future NDVI decline (p<0.05)"
                if p_value is not None and p_value < 0.05
                else "Infrastructure predicts future NDVI decline"
                if lag_corr < -0.3
                else "Weak delayed effect detected"
            ),
        })
        print(f"    [{zone}] lag_corr: {lag_corr:.4f} | p-value: {p_value}")

    granger_df = pd.DataFrame(results)
    granger_df.to_csv(os.path.join(PROC_DIR, "granger_results.csv"), index=False)
    print(f"    ✓ Granger saved → data/processed/granger_results.csv")
    return granger_df


def compute_clusters(df: pd.DataFrame, feature_cols: list):
    print("\n[+] K-Means degradation clusters...")

    X      = df[feature_cols].fillna(0)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df     = df.copy()
    df["cluster"] = kmeans.fit_predict(X)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
    cluster_names = {}
    for i in range(4):
        ndvi_val  = centers.iloc[i].get("ndvi_total_change", 0)
        infra_val = centers.iloc[i].get("infra_pressure", 0)
        rain_val  = centers.iloc[i].get("rainfall_anomaly", 0)
        if ndvi_val < -0.1:
            cluster_names[i] = "Vegetation-loss zone"
        elif infra_val > 0.3:
            cluster_names[i] = "Infrastructure-pressure zone"
        elif rain_val < -50:
            cluster_names[i] = "Climate-stressed zone"
        else:
            cluster_names[i] = "Ecologically stable zone"

    df["cluster_name"] = df["cluster"].map(cluster_names)

    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)

    for name, count in df["cluster_name"].value_counts().items():
        print(f"      {name}: {count} cells")

    return df


def run_ml_pipeline():
    print("=" * 60)
    print("  EcoShift AI — ML Models")
    print("=" * 60)

    df                          = load_data()
    model, le, feature_cols, df = train_model(df)
    shap_df, importance_df      = compute_shap(model, df, feature_cols)
    df, pca                     = compute_pca(df, feature_cols)
    granger_df                  = compute_granger(df)
    df                          = compute_clusters(df, feature_cols)

    out = os.path.join(PROC_DIR, "final_results.csv")
    df.to_csv(out, index=False)
    print(f"\n    ✓ Final results → {out}")

    print("\n" + "=" * 60)
    print("  ✓ ML pipeline complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_ml_pipeline()