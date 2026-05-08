"""
EcoShift AI — Module 2: Feature Engineering
Merges all raw data and engineers features for ML modeling.

Usage: python3 backend/feature_engineering.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# Bengaluru CBD coordinates
CBD_LAT = 12.9716
CBD_LON = 77.5946


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load all raw CSVs
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data():
    print("\n[1/5] Loading raw data...")

    grid     = pd.read_csv(os.path.join(RAW_DIR, "grid.csv"))
    ndvi     = pd.read_csv(os.path.join(RAW_DIR, "ndvi_grid.csv"))
    rainfall = pd.read_csv(os.path.join(RAW_DIR, "rainfall_grid.csv"))
    osm      = pd.read_csv(os.path.join(RAW_DIR, "osm_grid.csv"))
    gbif     = pd.read_csv(os.path.join(RAW_DIR, "gbif_grid.csv"))

    print(f"    grid: {len(grid)} rows")
    print(f"    ndvi: {len(ndvi)} rows")
    print(f"    rainfall: {len(rainfall)} rows")
    print(f"    osm: {len(osm)} rows")
    print(f"    gbif: {len(gbif)} rows")

    return grid, ndvi, rainfall, osm, gbif


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Merge all datasets on cell_id
# ─────────────────────────────────────────────────────────────────────────────
def merge_datasets(grid, ndvi, rainfall, osm, gbif) -> pd.DataFrame:
    print("\n[2/5] Merging datasets...")

    df = grid.copy()
    df = df.merge(ndvi,     on="cell_id", how="left")
    df = df.merge(rainfall, on="cell_id", how="left")
    df = df.merge(osm,      on="cell_id", how="left")
    df = df.merge(gbif,     on="cell_id", how="left")

    # Fill missing NDVI with median (spatial interpolation)
    ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
    for col in ndvi_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"    Filled {col} NaN with median: {median_val:.4f}")

    print(f"    ✓ Merged dataframe: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Engineer Features
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Engineering features...")

    # ── Distance from Bengaluru CBD (km) ──────────────────────────────────
    df["dist_to_cbd_km"] = np.sqrt(
        ((df["center_lat"] - CBD_LAT) * 111.0) ** 2 +
        ((df["center_lon"] - CBD_LON) * 111.0 *
         np.cos(np.radians(df["center_lat"]))) ** 2
    ).round(4)

    # ── Urban fringe zone ─────────────────────────────────────────────────
    # 0-10km = urban core, 10-25km = fringe, 25km+ = rural
    df["zone"] = pd.cut(
        df["dist_to_cbd_km"],
        bins=[0, 10, 25, 999],
        labels=["urban_core", "fringe", "rural"]
    ).astype(str)

    # ── NDVI-based features ───────────────────────────────────────────────
    ndvi_cols = ["ndvi_2018", "ndvi_2021", "ndvi_2024"]
    df["ndvi_mean"]        = df[ndvi_cols].mean(axis=1)
    df["ndvi_std"]         = df[ndvi_cols].std(axis=1)
    df["ndvi_min"]         = df[ndvi_cols].min(axis=1)

    # Acceleration of decline (is degradation speeding up?)
    df["ndvi_accel"] = (
        df["ndvi_change_2021_2024"] - df["ndvi_change_2018_2021"]
    )

    # Vegetation stress flag: total NDVI drop > 0.1
    df["vegetation_stressed"] = (df["ndvi_total_change"] < -0.1).astype(int)

    # ── Fragmentation index ───────────────────────────────────────────────
    # High std + low mean NDVI = fragmented patchy vegetation
    df["fragmentation_index"] = (
        df["ndvi_std"] / (df["ndvi_mean"].replace(0, np.nan) + 1e-6)
    ).round(4)

    # ── Infrastructure pressure score ─────────────────────────────────────
    road_norm  = df["road_length_km"].fillna(0)
    bldg_norm  = df["building_count"].fillna(0)
    road_max   = road_norm.max() if road_norm.max() > 0 else 1
    bldg_max   = bldg_norm.max() if bldg_norm.max() > 0 else 1

    df["infra_pressure"] = (
        0.5 * (road_norm / road_max) +
        0.5 * (bldg_norm / bldg_max)
    ).round(4)

    # ── Urban proximity pressure ──────────────────────────────────────────
    # Closer to CBD = higher pressure
    df["urban_proximity_score"] = (
        1 / (df["dist_to_cbd_km"] + 1)
    ).round(6)

    # ── Rainfall stress ───────────────────────────────────────────────────
    df["rainfall_stress"] = (
        df["rainfall_anomaly"].fillna(0) < -100
    ).astype(int)

    # ── Biodiversity density ──────────────────────────────────────────────
    df["biodiversity_density"] = (
        df["gbif_observations"].fillna(0) /
        (df["gbif_species_count"].fillna(1).replace(0, 1))
    ).round(4)

    # ── TEMPORAL LAG FEATURES ─────────────────────────────────────────────
    # Key insight: infrastructure pressure now predicts NDVI loss later
    # infra_pressure (current) is a proxy for past development
    # We model: high infra + prior NDVI decline = future collapse risk

    df["lag_stress_score"] = (
        df["infra_pressure"] * 0.4 +
        (df["ndvi_change_2015_2018"].fillna(0).clip(-1, 0).abs()) * 0.3 +
        (df["ndvi_change_2018_2021"].fillna(0).clip(-1, 0).abs()) * 0.3
    ).round(4)

    # ── Combined Ecological Vulnerability Score (EVS) ─────────────────────
    # Composite: higher = more vulnerable
    ndvi_loss   = df["ndvi_total_change"].fillna(0).clip(-1, 0).abs()
    ndvi_loss_n = ndvi_loss / (ndvi_loss.max() + 1e-6)
    frag_n      = df["fragmentation_index"] / (df["fragmentation_index"].max() + 1e-6)
    infra_n     = df["infra_pressure"]
    urb_n       = df["urban_proximity_score"] / (df["urban_proximity_score"].max() + 1e-6)

    df["ecological_vulnerability_score"] = (
        0.35 * ndvi_loss_n +
        0.25 * frag_n +
        0.25 * infra_n +
        0.15 * urb_n
    ).round(4)

    # ── Target Label ──────────────────────────────────────────────────────
    # Classify zones for supervised learning
    evs = df["ecological_vulnerability_score"]
    df["degradation_label"] = pd.cut(
        evs,
        bins=[0, 0.33, 0.66, 1.01],
        labels=["stable", "at_risk", "degraded"]
    ).astype(str)

    print(f"    ✓ Features engineered: {df.shape[1]} total columns")
    print(f"    Label distribution:\n{df['degradation_label'].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/5] Handling missing values...")

    # NDVI: fill with column median
    ndvi_cols = [c for c in df.columns if "ndvi" in c and c != "ndvi_total_change"]
    for col in ndvi_cols:
        df[col] = df[col].fillna(df[col].median())

    # Rainfall: fill with mean
    rain_cols = [c for c in df.columns if "rainfall" in c]
    for col in rain_cols:
        df[col] = df[col].fillna(df[col].mean())

    # OSM/GBIF: fill with 0
    for col in ["road_length_km", "building_count",
                "gbif_observations", "gbif_species_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Derived features: fill with 0
    derived = ["fragmentation_index", "infra_pressure",
               "lag_stress_score", "ecological_vulnerability_score",
               "biodiversity_density", "rainfall_anomaly"]
    for col in derived:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    missing_pct = (df.isnull().sum().sum() / df.size * 100).round(2)
    print(f"    ✓ Missing values remaining: {missing_pct}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_matrix(df: pd.DataFrame):
    print("\n[5/5] Saving feature matrix...")

    path = os.path.join(PROC_DIR, "feature_matrix.csv")
    df.to_csv(path, index=False)
    print(f"    ✓ Feature matrix saved  →  {path}")
    print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Also save just the ML feature columns for quick model loading
    feature_cols = [
        "cell_id", "center_lat", "center_lon",
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
        "ecological_vulnerability_score",
        "degradation_label",
    ]

    existing_cols = [c for c in feature_cols if c in df.columns]
    ml_df = df[existing_cols]
    ml_path = os.path.join(PROC_DIR, "ml_features.csv")
    ml_df.to_csv(ml_path, index=False)
    print(f"    ✓ ML features saved  →  {ml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_feature_engineering():
    print("=" * 60)
    print("  EcoShift AI — Feature Engineering")
    print("=" * 60)

    grid, ndvi, rainfall, osm, gbif = load_raw_data()
    df = merge_datasets(grid, ndvi, rainfall, osm, gbif)
    df = engineer_features(df)
    df = handle_missing(df)
    save_feature_matrix(df)

    print("\n" + "=" * 60)
    print("  ✓ Feature engineering complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_feature_engineering()