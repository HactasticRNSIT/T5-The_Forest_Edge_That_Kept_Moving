# 🌿 EcoShift AI
### Explainable Spatio-Temporal Ecological Intelligence Platform

> *"Transforming fragmented environmental signals into explainable ecological intelligence capable of predicting and explaining how urban development reshapes ecosystem boundaries over time."*

---

## Problem Statement
**The Forest Edge That Kept Moving** — Understanding why ecological boundaries shift unevenly across nearby regions under urban development pressure.

EcoShift AI goes beyond detecting visible deforestation. It uncovers **latent dependencies**, **delayed environmental effects**, and **hidden stress patterns** that explain why some ecosystems collapse while neighboring zones remain resilient.

---

## Live Demo
Start the backend and open the dashboard:
```bash
python3 -m uvicorn backend.api:app --reload --port 8000
# Open http://localhost:8000
```

---

## Architecture

```
Data Sources → Pipeline → Feature Engineering → ML Models → FastAPI → Dashboard
```

### Data Sources (4 real-time feeds)
| Source | Data | Coverage |
|--------|------|----------|
| Sentinel-2 / Landsat (Planetary Computer) | NDVI vegetation index | 2015–2024 |
| Open-Meteo API | Rainfall history | 2015–2024 |
| OpenStreetMap (osmnx) | Road network + buildings | Current |
| GBIF API | Biodiversity observations | 2015–2024 |

### ML Pipeline (5 engines)
| Engine | Purpose |
|--------|---------|
| Random Forest | Classify zones: stable / at-risk / degraded |
| SHAP | Explain WHY each zone is degrading |
| PCA | Extract latent stress axes (hidden factors) |
| K-Means | Cluster zones by degradation fingerprint |
| Granger Lag Analysis | Prove delayed effects: infra at T → NDVI loss at T+2 |

---

## Key Results
- **3,750 grid cells** analyzed across Bengaluru urban fringe
- **10 years** of satellite + environmental data (2015–2024)
- **22 features** engineered per zone
- **Mean NDVI change: -0.33** — real vegetation decline detected
- **20 at-risk zones** identified with orange warning classification
- **1 latent stress zone** — appears stable but predicted to degrade within 2–3 years

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for React frontend)

### Install dependencies
```bash
pip3 install -r requirements.txt
```

### Run data pipeline (first time only)
```bash
# Fetch NDVI from Sentinel-2
python3 backend/ndvi_fetcher.py

# Fetch rainfall, OSM, GBIF
python3 backend/data_pipeline.py

# Engineer features
python3 backend/feature_engineering.py

# Train ML models
python3 backend/ml_models.py
```

### Start the application
```bash
# Terminal 1 — Backend API
python3 -m uvicorn backend.api:app --reload --port 8000

# Terminal 2 — React frontend (optional)
cd frontend && npm install && npm run dev
```

Open `http://localhost:8000` for the full dashboard.

---

## Project Structure
```
ECOSHIFT-AI/
├── backend/
│   ├── api.py                  # FastAPI REST endpoints
│   ├── data_pipeline.py        # Multi-source data fetcher
│   ├── ndvi_fetcher.py         # Sentinel-2 NDVI via Planetary Computer
│   ├── feature_engineering.py  # Feature matrix builder
│   └── ml_models.py            # RF + SHAP + PCA + Granger + KMeans
├── data/
│   ├── raw/                    # Downloaded CSVs
│   └── processed/              # Feature matrix + ML outputs
├── frontend/
│   ├── index.html              # Main dashboard (HTML/CSS/JS)
│   └── src/                    # React components
├── models/                     # Saved .pkl model files
├── docs/
├── requirements.txt
└── README.md
```

---

## API Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /` | Serve dashboard |
| `GET /api/stats` | Overall KPI statistics |
| `GET /api/zones` | Grid cells with coordinates + labels |
| `GET /api/importance` | SHAP feature importance |
| `GET /api/granger` | Delayed effect analysis |
| `GET /api/clusters` | K-Means degradation clusters |
| `GET /api/ndvi_timeline` | NDVI over time by zone type |
| `GET /api/zone/{cell_id}` | Detail for specific grid cell |

---

## Dashboard Features
- 🗺 **Interactive Risk Map** — 3,750 color-coded zones over Bengaluru
- 📈 **NDVI Timeline** — Vegetation health trend 2015→2024
- 🔍 **SHAP Drivers** — Top factors causing degradation
- 🧬 **Degradation Fingerprints** — K-Means cluster breakdown
- ⚡ **Latent Stress Score** — Hidden pressure zones
- ⏳ **Delayed Effect Panel** — Infrastructure → vegetation lag

---

## Tech Stack
`Python` · `FastAPI` · `Uvicorn` · `Pandas` · `NumPy` · `GeoPandas` · `Rasterio` · `scikit-learn` · `SHAP` · `pystac-client` · `planetary-computer` · `osmnx` · `HTML/CSS/JS` · `Leaflet.js` · `React` · `Vite`

---

## Study Area
**Bengaluru Urban Fringe** — 12.70–13.20°N, 77.30–77.90°E

Bengaluru is one of India's fastest-growing cities, making it an ideal case study for urban-ecological boundary dynamics. The study area covers the transition zones between the city core and surrounding forests, wetlands, and agricultural land.

---

*Built for Hackathon 2025 — Problem Statement 15: The Forest Edge That Kept Moving*