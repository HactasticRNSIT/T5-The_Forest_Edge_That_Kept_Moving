"""
EcoShift AI — Module 1: Data Pipeline
Fetches and saves all raw data for Bengaluru urban fringe analysis.
Uses Microsoft Planetary Computer for NDVI — NO account needed.

Run ONCE to populate data/raw/
Usage: python3 backend/data_pipeline.py
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from shapely.geometry import box
from tqdm import tqdm
import pystac_client
import planetary_computer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BBOX = {
    "min_lat": 12.70,
    "max_lat": 13.20,
    "min_lon": 77.30,
    "max_lon": 77.90,
}

GRID_SIZE_DEG = 0.009
YEARS         = [2015, 2018, 2021, 2024]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build Grid
# ─────────────────────────────────────────────────────────────────────────────
def build_grid() -> gpd.GeoDataFrame:
    print("\n[1/5] Building 1km grid over Bengaluru...")
    lats = np.arange(BBOX["min_lat"], BBOX["max_lat"], GRID_SIZE_DEG)
    lons = np.arange(BBOX["min_lon"], BBOX["max_lon"], GRID_SIZE_DEG)

    cells = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            cells.append({
                "cell_id":    cell_id,
                "center_lat": round(lat + GRID_SIZE_DEG / 2, 6),
                "center_lon": round(lon + GRID_SIZE_DEG / 2, 6),
                "geometry":   box(lon, lat,
                                  lon + GRID_SIZE_DEG,
                                  lat + GRID_SIZE_DEG),
            })
            cell_id += 1

    gdf  = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    path = os.path.join(RAW_DIR, "grid.csv")
    gdf[["cell_id", "center_lat", "center_lon"]].to_csv(path, index=False)
    print(f"    ✓ {len(gdf)} grid cells  →  {path}")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — NDVI via Sentinel-2 on Planetary Computer (no account needed)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ndvi(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n[2/5] Fetching NDVI from Sentinel-2 (Planetary Computer)...")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    bbox_list = [
        BBOX["min_lon"], BBOX["min_lat"],
        BBOX["max_lon"], BBOX["max_lat"],
    ]

    all_records = {
        int(row.cell_id): {"cell_id": int(row.cell_id)}
        for _, row in grid.iterrows()
    }

    for year in YEARS:
        print(f"    Searching Sentinel-2 for {year}...")
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox_list,
                datetime=f"{year}-06-01/{year}-09-30",
                query={"eo:cloud_cover": {"lt": 20}},
            )
            items = list(search.get_items())
        except Exception as e:
            print(f"    [WARN] Search failed for {year}: {e}")
            items = []

        if not items:
            # fallback: try Landsat
            print(f"    Sentinel-2 empty for {year}, trying Landsat...")
            try:
                search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=bbox_list,
                    datetime=f"{year}-06-01/{year}-09-30",
                    query={
                        "eo:cloud_cover": {"lt": 20},
                        "platform": {"in": ["landsat-8", "landsat-9"]},
                    },
                )
                items = list(search.get_items())
                source = "landsat"
            except Exception:
                items  = []
                source = "none"
        else:
            source = "sentinel"

        if not items:
            print(f"    [WARN] No scenes for {year} — filling NaN")
            for cid in all_records:
                all_records[cid][f"ndvi_{year}"] = np.nan
            continue

        item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        item = planetary_computer.sign(item)
        print(f"    [{source}] {item.id} "
              f"(cloud={item.properties.get('eo:cloud_cover','?')}%)")

        # Band names differ between Sentinel-2 and Landsat
        if source == "sentinel":
            nir_key, red_key = "B08", "B04"
            scale, offset    = 1 / 10000.0, 0.0
        else:
            nir_key, red_key = "nir08", "red"
            scale, offset    = 0.0000275, -0.2

        nir_url = item.assets[nir_key].href
        red_url = item.assets[red_key].href

        try:
            with rasterio.open(nir_url) as nir_src, \
                 rasterio.open(red_url) as red_src:

                win_nir = from_bounds(
                    BBOX["min_lon"], BBOX["min_lat"],
                    BBOX["max_lon"], BBOX["max_lat"],
                    transform=nir_src.transform,
                )
                win_red = from_bounds(
                    BBOX["min_lon"], BBOX["min_lat"],
                    BBOX["max_lon"], BBOX["max_lat"],
                    transform=red_src.transform,
                )

                nir_arr = nir_src.read(1, window=win_nir).astype(np.float32)
                red_arr = red_src.read(1, window=win_red).astype(np.float32)

                nir_arr = nir_arr * scale + offset
                red_arr = red_arr * scale + offset
                nir_arr[nir_arr <= 0] = np.nan
                red_arr[red_arr <= 0] = np.nan

                ndvi_arr      = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-10)
                nir_transform = nir_src.window_transform(win_nir)

            for _, row in grid.iterrows():
                try:
                    r, c = rowcol(nir_transform,
                                  row.center_lon, row.center_lat)
                    if 0 <= r < ndvi_arr.shape[0] and \
                       0 <= c < ndvi_arr.shape[1]:
                        val = float(ndvi_arr[r, c])
                    else:
                        val = np.nan
                except Exception:
                    val = np.nan
                all_records[int(row.cell_id)][f"ndvi_{year}"] = round(val, 4)

        except Exception as e:
            print(f"    [WARN] Raster read failed for {year}: {e}")
            for cid in all_records:
                all_records[cid][f"ndvi_{year}"] = np.nan

    df = pd.DataFrame(list(all_records.values()))

    df["ndvi_change_2015_2018"] = df["ndvi_2018"] - df["ndvi_2015"]
    df["ndvi_change_2018_2021"] = df["ndvi_2021"] - df["ndvi_2018"]
    df["ndvi_change_2021_2024"] = df["ndvi_2024"] - df["ndvi_2021"]
    df["ndvi_total_change"]     = df["ndvi_2024"] - df["ndvi_2015"]

    path = os.path.join(RAW_DIR, "ndvi_grid.csv")
    df.to_csv(path, index=False)
    print(f"    ✓ NDVI saved  →  {path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Rainfall via Open-Meteo (free, no auth)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_rainfall(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n[3/5] Fetching rainfall from Open-Meteo...")

    sample   = grid.iloc[::5].copy().reset_index(drop=True)
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    records  = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="    Rainfall"):
        params = {
            "latitude":   row.center_lat,
            "longitude":  row.center_lon,
            "start_date": "2015-01-01",
            "end_date":   "2024-12-31",
            "daily":      "precipitation_sum",
            "timezone":   "Asia/Kolkata",
        }
        try:
            resp  = requests.get(BASE_URL, params=params, timeout=15)
            data  = resp.json()
            daily = pd.DataFrame({
                "date": data["daily"]["time"],
                "rain": data["daily"]["precipitation_sum"],
            })
            daily["date"] = pd.to_datetime(daily["date"])
            daily["year"] = daily["date"].dt.year
            annual = daily.groupby("year")["rain"].sum().reset_index()
            for _, yr in annual.iterrows():
                records.append({
                    "cell_id":            int(row.cell_id),
                    "year":               int(yr["year"]),
                    "annual_rainfall_mm": round(yr["rain"], 2),
                })
        except Exception:
            pass
        time.sleep(0.05)

    df      = pd.DataFrame(records)
    df_wide = df.pivot(
        index="cell_id", columns="year",
        values="annual_rainfall_mm"
    ).reset_index()
    df_wide.columns = [
        f"rainfall_{c}" if isinstance(c, int) else c
        for c in df_wide.columns
    ]

    rain_cols = [c for c in df_wide.columns if c.startswith("rainfall_")]
    df_wide["rainfall_mean"]    = df_wide[rain_cols].mean(axis=1)
    df_wide["rainfall_anomaly"] = (
        df_wide.get("rainfall_2024", np.nan) - df_wide["rainfall_mean"]
    )

    path = os.path.join(RAW_DIR, "rainfall_grid.csv")
    df_wide.to_csv(path, index=False)
    print(f"    ✓ Rainfall saved  →  {path}  ({len(df_wide)} rows)")
    return df_wide


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Infrastructure via OpenStreetMap (fixed for osmnx 2.x)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_osm_infrastructure(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n[4/5] Fetching infrastructure from OSM...")

    bbox_tuple = (
        BBOX["max_lat"], BBOX["min_lat"],
        BBOX["max_lon"], BBOX["min_lon"]
    )

    print("    Downloading road network (may take ~2 min)...")
    try:
        G         = ox.graph_from_bbox(bbox_tuple, network_type="drive")
        roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)[1]
        roads_gdf = roads_gdf.to_crs("EPSG:32643")
        print(f"    ✓ Roads downloaded: {len(roads_gdf)} edges")
    except Exception as e:
        print(f"    [WARN] Roads failed: {e}")
        roads_gdf = None

    print("    Downloading building footprints...")
    try:
        buildings_gdf = ox.features_from_bbox(
            bbox_tuple, tags={"building": True}
        )
        buildings_gdf = buildings_gdf.to_crs("EPSG:32643")
        print(f"    ✓ Buildings downloaded: {len(buildings_gdf)} features")
    except Exception as e:
        print(f"    [WARN] Buildings failed: {e}")
        buildings_gdf = None

    grid_utm = grid.to_crs("EPSG:32643")
    records  = []

    for _, cell in tqdm(grid_utm.iterrows(), total=len(grid_utm),
                        desc="    OSM per cell"):
        geom     = cell.geometry
        road_len = 0.0
        bldg_cnt = 0

        if roads_gdf is not None:
            try:
                road_len = (
                    roads_gdf[roads_gdf.intersects(geom)].length.sum() / 1000
                )
            except Exception:
                pass

        if buildings_gdf is not None:
            try:
                bldg_cnt = len(buildings_gdf[buildings_gdf.intersects(geom)])
            except Exception:
                pass

        records.append({
            "cell_id":        int(cell.cell_id),
            "road_length_km": round(road_len, 4),
            "building_count": bldg_cnt,
        })

    df   = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "osm_grid.csv")
    df.to_csv(path, index=False)
    print(f"    ✓ OSM saved  →  {path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Biodiversity via GBIF (free, no auth)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_gbif_biodiversity(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n[5/5] Fetching biodiversity from GBIF...")

    BASE_URL = "https://api.gbif.org/v1/occurrence/search"
    sample   = grid.iloc[::10].copy().reset_index(drop=True)
    half     = GRID_SIZE_DEG / 2
    records  = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="    GBIF"):
        params = {
            "decimalLatitude":  f"{row.center_lat - half},{row.center_lat + half}",
            "decimalLongitude": f"{row.center_lon - half},{row.center_lon + half}",
            "hasCoordinate":    "true",
            "limit":            300,
            "year":             "2015,2024",
        }
        try:
            resp    = requests.get(BASE_URL, params=params, timeout=15)
            data    = resp.json()
            results = data.get("results", [])
            species = len(set(
                r.get("species", "") for r in results if r.get("species")
            ))
            records.append({
                "cell_id":            int(row.cell_id),
                "gbif_observations":  data.get("count", 0),
                "gbif_species_count": species,
            })
        except Exception:
            pass
        time.sleep(0.2)

    df   = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "gbif_grid.csv")
    df.to_csv(path, index=False)
    print(f"    ✓ GBIF saved  →  {path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  EcoShift AI — Data Pipeline")
    print("=" * 60)

    grid = build_grid()
    fetch_ndvi(grid)
    fetch_rainfall(grid)
    fetch_osm_infrastructure(grid)
    fetch_gbif_biodiversity(grid)

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete. Files saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()  