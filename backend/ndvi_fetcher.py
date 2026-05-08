"""
EcoShift AI — NDVI Fetcher (Fixed)
Fetches real Sentinel-2 NDVI for Bengaluru using Planetary Computer.
Has robust error handling and validation.

Usage: python3 backend/ndvi_fetcher.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import pystac_client
import planetary_computer

warnings.filterwarnings("ignore")

BBOX = {
    "min_lat": 12.70,
    "max_lat": 13.20,
    "min_lon": 77.30,
    "max_lon": 77.90,
}

YEARS         = [2015, 2018, 2021, 2024]
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


def fetch_ndvi_for_year(catalog, year: int, grid: gpd.GeoDataFrame) -> dict:
    """Fetch NDVI for a single year. Returns {cell_id: ndvi_value}."""
    bbox_list = [BBOX["min_lon"], BBOX["min_lat"], BBOX["max_lon"], BBOX["max_lat"]]

    # Try Sentinel-2 first
    print(f"    Searching Sentinel-2 for {year}...")
    items = []
    source = None

    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox_list,
            datetime=f"{year}-06-01/{year}-09-30",
            query={"eo:cloud_cover": {"lt": 30}},
        )
        items = list(search.get_items())
        if items:
            source = "sentinel-2"
            print(f"    Found {len(items)} Sentinel-2 scenes for {year}")
    except Exception as e:
        print(f"    Sentinel-2 search failed: {e}")

    # Fallback to Landsat
    if not items:
        print(f"    Trying Landsat for {year}...")
        try:
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox_list,
                datetime=f"{year}-06-01/{year}-09-30",
                query={
                    "eo:cloud_cover": {"lt": 30},
                    "platform": {"in": ["landsat-8", "landsat-9"]},
                },
            )
            items = list(search.get_items())
            if items:
                source = "landsat"
                print(f"    Found {len(items)} Landsat scenes for {year}")
        except Exception as e:
            print(f"    Landsat search failed: {e}")

    if not items:
        print(f"    [WARN] No scenes found for {year}")
        return {}

    # Pick least cloudy scene
    item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = planetary_computer.sign(item)
    cloud = item.properties.get("eo:cloud_cover", "?")
    print(f"    Using {source} scene: {item.id} (cloud={cloud}%)")

    # Get band URLs
    if source == "sentinel-2":
        nir_key, red_key = "B08", "B04"
        scale, offset = 1 / 10000.0, 0.0
    else:
        nir_key, red_key = "nir08", "red"
        scale, offset = 0.0000275, -0.2

    if nir_key not in item.assets or red_key not in item.assets:
        print(f"    [WARN] Missing bands in scene for {year}")
        return {}

    nir_url = item.assets[nir_key].href
    red_url = item.assets[red_key].href

    try:
        with rasterio.open(nir_url) as nir_src:
            # Get the CRS of the raster
            raster_crs = nir_src.crs

            # Reproject bbox to raster CRS
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds
            bounds_in_crs = transform_bounds(
                "EPSG:4326", raster_crs,
                BBOX["min_lon"], BBOX["min_lat"],
                BBOX["max_lon"], BBOX["max_lat"]
            )

            win_nir = from_bounds(*bounds_in_crs, transform=nir_src.transform)
            nir_arr = nir_src.read(1, window=win_nir).astype(np.float32)
            nir_transform = nir_src.window_transform(win_nir)

        with rasterio.open(red_url) as red_src:
            raster_crs_red = red_src.crs
            bounds_red = transform_bounds(
                "EPSG:4326", raster_crs_red,
                BBOX["min_lon"], BBOX["min_lat"],
                BBOX["max_lon"], BBOX["max_lat"]
            )
            win_red = from_bounds(*bounds_red, transform=red_src.transform)
            red_arr = red_src.read(1, window=win_red).astype(np.float32)

        # Apply scale factors
        nir_arr = nir_arr * scale + offset
        red_arr = red_arr * scale + offset

        # Mask invalid values
        nir_arr[nir_arr <= 0] = np.nan
        nir_arr[nir_arr > 1] = np.nan
        red_arr[red_arr <= 0] = np.nan
        red_arr[red_arr > 1] = np.nan

        # Resize red to match nir if shapes differ
        if nir_arr.shape != red_arr.shape:
            from PIL import Image
            red_img = Image.fromarray(red_arr)
            red_img = red_img.resize((nir_arr.shape[1], nir_arr.shape[0]))
            red_arr = np.array(red_img)

        # Compute NDVI
        ndvi_arr = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-10)
        ndvi_arr = np.clip(ndvi_arr, -1, 1)

        # Validate
        valid_pct = np.sum(~np.isnan(ndvi_arr)) / ndvi_arr.size * 100
        print(f"    NDVI array shape: {ndvi_arr.shape}, valid pixels: {valid_pct:.1f}%")

        if valid_pct < 5:
            print(f"    [WARN] Too few valid pixels for {year}")
            return {}

        mean_ndvi = np.nanmean(ndvi_arr)
        print(f"    Mean NDVI for {year}: {mean_ndvi:.4f}")

        # Sample NDVI at each grid cell centroid
        results = {}
        from rasterio.warp import transform as warp_transform

        for _, row in grid.iterrows():
            try:
                # Transform lat/lon to raster CRS
                xs, ys = warp_transform(
                    "EPSG:4326", raster_crs,
                    [row.center_lon], [row.center_lat]
                )
                r, c = rowcol(nir_transform, xs[0], ys[0])
                if 0 <= r < ndvi_arr.shape[0] and 0 <= c < ndvi_arr.shape[1]:
                    val = float(ndvi_arr[r, c])
                    if not np.isnan(val):
                        results[int(row.cell_id)] = round(val, 4)
            except Exception:
                pass

        print(f"    Sampled NDVI for {len(results)} cells")
        return results

    except Exception as e:
        print(f"    [ERROR] Failed reading raster for {year}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_ndvi_fetcher():
    print("=" * 60)
    print("  EcoShift AI — NDVI Fetcher (Fixed)")
    print("=" * 60)

    print("\nConnecting to Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    print("✓ Connected")

    print("\nBuilding grid...")
    grid = build_grid()
    print(f"✓ {len(grid)} grid cells")

    # Fetch NDVI per year
    all_ndvi = {int(row.cell_id): {"cell_id": int(row.cell_id)}
                for _, row in grid.iterrows()}

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        year_results = fetch_ndvi_for_year(catalog, year, grid)

        if year_results:
            for cell_id, val in year_results.items():
                all_ndvi[cell_id][f"ndvi_{year}"] = val
            print(f"✓ NDVI {year} complete: {len(year_results)} cells")
        else:
            print(f"[WARN] No NDVI data for {year} — filling with NaN")
            for cell_id in all_ndvi:
                all_ndvi[cell_id][f"ndvi_{year}"] = np.nan

    # Build dataframe
    df = pd.DataFrame(list(all_ndvi.values()))

    # Compute change rates
    for col in [f"ndvi_{y}" for y in YEARS]:
        if col not in df.columns:
            df[col] = np.nan

    df["ndvi_change_2015_2018"] = df["ndvi_2018"] - df["ndvi_2015"]
    df["ndvi_change_2018_2021"] = df["ndvi_2021"] - df["ndvi_2018"]
    df["ndvi_change_2021_2024"] = df["ndvi_2024"] - df["ndvi_2021"]
    df["ndvi_total_change"]     = df["ndvi_2024"] - df["ndvi_2015"]

    # Validate
    valid_rows = df["ndvi_2024"].notna().sum()
    print(f"\n✓ Valid NDVI 2024 values: {valid_rows}/{len(df)} cells")
    print(f"  Mean NDVI 2024: {df['ndvi_2024'].mean():.4f}")
    print(f"  Mean total change: {df['ndvi_total_change'].mean():.4f}")

    # Save
    out_path = os.path.join(RAW_DIR, "ndvi_grid.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✓ NDVI saved → {out_path}")
    print("=" * 60)
    print("  NDVI fetch complete. Now run:")
    print("  python3 backend/feature_engineering.py")
    print("  python3 backend/ml_models.py")
    print("=" * 60)

    return df


if __name__ == "__main__":
    run_ndvi_fetcher()