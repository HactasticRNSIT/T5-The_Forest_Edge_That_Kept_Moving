"""
EcoShift AI — OSM Fetcher (Fixed for osmnx 2.1.0)
Fetches road network and building data for Bengaluru.

Usage: python3 backend/osm_fetcher.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
from tqdm import tqdm

warnings.filterwarnings("ignore")

BBOX = {
    "min_lat": 12.70,
    "max_lat": 13.20,
    "min_lon": 77.30,
    "max_lon": 77.90,
}

GRID_SIZE_DEG = 0.009

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def build_grid() -> gpd.GeoDataFrame:
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
                "geometry":   box(lon, lat, lon + GRID_SIZE_DEG, lat + GRID_SIZE_DEG),
            })
            cell_id += 1
    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def fetch_osm_infrastructure(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\nFetching OSM infrastructure (osmnx 2.1.0)...")

    # osmnx 2.x uses positional bbox as (left, bottom, right, top) = (west, south, east, north)
    west  = BBOX["min_lon"]
    south = BBOX["min_lat"]
    east  = BBOX["max_lon"]
    north = BBOX["max_lat"]

    # Roads
    print("    Downloading road network...")
    roads_gdf = None
    try:
        # osmnx 2.x: graph_from_bbox(bbox) where bbox = (left, bottom, right, top)
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type="drive"
        )
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        roads_gdf = edges.to_crs("EPSG:32643")
        print(f"    ✓ Roads: {len(roads_gdf)} edges")
    except Exception as e:
        print(f"    [WARN] Roads v1 failed: {e}")
        try:
            # Alternative API
            G = ox.graph_from_bbox(
                north=north, south=south, east=east, west=west,
                network_type="drive"
            )
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads_gdf = edges.to_crs("EPSG:32643")
            print(f"    ✓ Roads (alt): {len(roads_gdf)} edges")
        except Exception as e2:
            print(f"    [WARN] Roads v2 failed: {e2}")

    # Buildings
    print("    Downloading building footprints...")
    buildings_gdf = None
    try:
        buildings_gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={"building": True}
        )
        buildings_gdf = buildings_gdf.to_crs("EPSG:32643")
        print(f"    ✓ Buildings: {len(buildings_gdf)} features")
    except Exception as e:
        print(f"    [WARN] Buildings v1 failed: {e}")
        try:
            buildings_gdf = ox.features_from_bbox(
                north=north, south=south, east=east, west=west,
                tags={"building": True}
            )
            buildings_gdf = buildings_gdf.to_crs("EPSG:32643")
            print(f"    ✓ Buildings (alt): {len(buildings_gdf)} features")
        except Exception as e2:
            print(f"    [WARN] Buildings v2 failed: {e2}")

    if roads_gdf is None and buildings_gdf is None:
        print("    [ERROR] Both roads and buildings failed")
        return pd.DataFrame()

    # Compute per-cell metrics
    grid_utm = grid.to_crs("EPSG:32643")
    records  = []

    print(f"    Computing metrics for {len(grid_utm)} cells...")
    for _, cell in tqdm(grid_utm.iterrows(), total=len(grid_utm), desc="    OSM"):
        geom     = cell.geometry
        road_len = 0.0
        bldg_cnt = 0

        if roads_gdf is not None:
            try:
                clipped  = roads_gdf[roads_gdf.intersects(geom)]
                road_len = float(clipped.length.sum() / 1000)
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

    df = pd.DataFrame(records)

    # Validate
    nonzero = (df["road_length_km"] > 0).sum()
    print(f"    ✓ Cells with road data: {nonzero}/{len(df)}")
    print(f"    Max road length: {df['road_length_km'].max():.3f} km")
    print(f"    Max buildings: {df['building_count'].max()}")

    path = os.path.join(RAW_DIR, "osm_grid.csv")
    df.to_csv(path, index=False)
    print(f"    ✓ OSM saved → {path}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  EcoShift AI — OSM Fetcher (osmnx 2.1.0)")
    print("=" * 60)

    grid = build_grid()
    print(f"Grid: {len(grid)} cells")

    df = fetch_osm_infrastructure(grid)

    if not df.empty:
        print("\n✓ OSM complete. Now run:")
        print("  python3 backend/feature_engineering.py")
        print("  python3 backend/ml_models.py")
    else:
        print("\n[ERROR] OSM fetch failed completely")