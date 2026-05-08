"""
EcoShift AI — FastAPI Backend
Serves ML results as JSON endpoints for the frontend

Usage: python3 -m uvicorn backend.api:app --reload --port 8000
"""

import os
import pickle
import warnings
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(__file__)
PROC_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
FRONT_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = FastAPI(title="EcoShift AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache dataframes on startup ───────────────────────────────────────────────
_df         = None
_importance = None
_granger    = None


def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(os.path.join(PROC_DIR, "final_results.csv"))
        _df = _df.fillna(0)
    return _df


def get_importance():
    global _importance
    if _importance is None:
        _importance = pd.read_csv(os.path.join(PROC_DIR, "feature_importance.csv"))
    return _importance


def get_granger():
    global _granger
    if _granger is None:
        path = os.path.join(PROC_DIR, "granger_results.csv")
        _granger = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    return _granger


# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONT_DIR, "index.html"))
# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """Overall KPI statistics"""
    df    = get_df()
    label = "predicted_label" if "predicted_label" in df.columns else "degradation_label"

    total    = len(df)
    degraded = int((df[label] == "degraded").sum())
    at_risk  = int((df[label] == "at_risk").sum())
    stable   = int((df[label] == "stable").sum())

    avg_ndvi    = round(float(df["ndvi_total_change"].mean()), 4)
    avg_stress  = round(float(df["latent_stress_norm"].mean()), 4) if "latent_stress_norm" in df.columns else 0
    avg_infra   = round(float(df["infra_pressure"].mean()), 4)
    avg_conf    = round(float(df["pred_confidence"].mean()), 4) if "pred_confidence" in df.columns else 0

    high_stress = int((df["latent_stress_norm"] > 0.7).sum()) if "latent_stress_norm" in df.columns else 0

    return {
        "total_cells":      total,
        "degraded":         degraded,
        "at_risk":          at_risk,
        "stable":           stable,
        "degraded_pct":     round(degraded / total * 100, 1),
        "at_risk_pct":      round(at_risk  / total * 100, 1),
        "stable_pct":       round(stable   / total * 100, 1),
        "avg_ndvi_change":  avg_ndvi,
        "avg_latent_stress": avg_stress,
        "avg_infra_pressure": avg_infra,
        "model_accuracy":   avg_conf,
        "high_stress_zones": high_stress,
    }


@app.get("/api/zones")
def get_zones(limit: int = 1000):
    """Grid cells with coordinates and labels for map rendering"""
    df    = get_df()
    label = "predicted_label" if "predicted_label" in df.columns else "degradation_label"

    cols = [
        "cell_id", "center_lat", "center_lon",
        label,
        "ndvi_total_change", "ndvi_2024",
        "infra_pressure", "latent_stress_norm",
        "pred_confidence", "cluster_name",
        "ecological_vulnerability_score",
        "fragmentation_index", "dist_to_cbd_km",
    ]
    existing = [c for c in cols if c in df.columns]
    sample   = df[existing].sample(min(limit, len(df)), random_state=42)

    # Rename label col for frontend consistency
    if label != "predicted_label":
        sample = sample.rename(columns={label: "predicted_label"})

    return sample.fillna(0).to_dict(orient="records")


@app.get("/api/importance")
def get_feature_importance():
    """SHAP feature importance"""
    imp = get_importance()
    return imp.head(12).to_dict(orient="records")


@app.get("/api/granger")
def get_granger_results():
    """Delayed effect analysis results"""
    gr = get_granger()
    if gr.empty:
        return []
    return gr.to_dict(orient="records")


@app.get("/api/clusters")
def get_clusters():
    """Cluster distribution"""
    df = get_df()
    if "cluster_name" not in df.columns:
        return []
    counts = df["cluster_name"].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    return counts.to_dict(orient="records")


@app.get("/api/ndvi_timeline")
def get_ndvi_timeline():
    """NDVI over time by degradation label"""
    df    = get_df()
    label = "predicted_label" if "predicted_label" in df.columns else "degradation_label"

    records = []
    for lbl in ["degraded", "at_risk", "stable"]:
        sub = df[df[label] == lbl]
        for year, col in [(2018, "ndvi_2018"), (2021, "ndvi_2021"), (2024, "ndvi_2024")]:
            if col in df.columns:
                records.append({
                    "year":   year,
                    "ndvi":   round(float(sub[col].mean()), 4),
                    "status": lbl,
                })
    return records


@app.get("/api/zone/{cell_id}")
def get_zone_detail(cell_id: int):
    """Detail for a specific grid cell"""
    df  = get_df()
    row = df[df["cell_id"] == cell_id]
    if row.empty:
        return {"error": "Cell not found"}

    imp      = get_importance()
    top_feat = imp.head(5)["feature"].tolist()

    result = row.iloc[0].fillna(0).to_dict()
    result["top_drivers"] = {
        f: round(float(row.iloc[0].get(f, 0)), 4)
        for f in top_feat if f in row.columns
    }
    return result