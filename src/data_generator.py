"""Semi-synthetic generator for paired Sentinel-2 indices, SoilGrids-style covariates and SOC.

The generator simulates the chain that produces a real field dataset, rather than
drawing features and target independently:

1. Latent spatial fields (elevation, rainfall, clay, pH, SOC residual) are sampled
   from approximate Gaussian processes, so neighbouring samples are correlated.
2. SOC (g/kg) follows a non-linear, heteroscedastic response to those fields.
3. Surface reflectance in Sentinel-2 B2, B4, B5 and B8 is produced by linear mixing
   of a vegetation and a bare-soil endmember. Soil brightness drops with SOC, which is
   the physical reason optical indices carry any SOC signal at all.
4. Observation error is applied last: radiometric noise, undetected haze/cloud
   residuals, SoilGrids map error on the tabular covariates, and lab error on SOC.
5. Sample locations are clustered (accessible sites) with a sparse background, which
   reproduces the uneven sampling density of legacy soil surveys.

Run as a script to write ``data/soc_synthetic.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ndvi",
    "evi",
    "ndre",
    "ph_h2o",
    "clay_pct",
    "elevation_m",
    "map_mm",
]
TARGET_COLUMN = "soc_g_kg"


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int = 1500
    lat_min: float = 0.0
    lat_max: float = 1.5
    lon_min: float = 31.5
    lon_max: float = 33.5
    # 0.25 deg (~28 km) is close to the range of the SOC residual field below, so
    # samples in different blocks are only weakly correlated.
    block_size_deg: float = 0.25
    n_clusters: int = 14
    cluster_fraction: float = 0.8
    cluster_sd_deg: float = 0.05
    # Share of acquisitions with haze or thin cloud that passed the L2A scene
    # classification mask. These are the residuals that reach the feature table.
    cloud_residual_prob: float = 0.10
    seed: int = 42


class _RandomField:
    """Stationary Gaussian field approximated with random Fourier features.

    A dense GP draw needs an O(n^3) Cholesky; RFF with a few hundred frequencies
    gives the same squared-exponential covariance to good accuracy, and the field
    is a deterministic function of (lat, lon) so it can be evaluated anywhere.
    """

    def __init__(self, rng: np.random.Generator, lengthscale_deg: float, n_features: int = 400):
        self.omega = rng.normal(0.0, 1.0 / lengthscale_deg, size=(n_features, 2))
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        self.weights = rng.normal(0.0, 1.0, size=n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        coords = np.column_stack([lat, lon])
        phi = self.scale * np.cos(coords @ self.omega.T + self.phase)
        return phi @ self.weights  # approximately unit variance


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_locations(cfg: GeneratorConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Clustered sampling design plus a sparse uniform background."""
    n_clustered = int(round(cfg.n_samples * cfg.cluster_fraction))
    n_background = cfg.n_samples - n_clustered

    centres_lat = rng.uniform(cfg.lat_min, cfg.lat_max, cfg.n_clusters)
    centres_lon = rng.uniform(cfg.lon_min, cfg.lon_max, cfg.n_clusters)
    # Unequal cluster sizes: survey effort is rarely balanced across districts.
    cluster_weights = rng.dirichlet(np.full(cfg.n_clusters, 0.8))
    assignment = rng.choice(cfg.n_clusters, size=n_clustered, p=cluster_weights)

    lat = np.concatenate([
        centres_lat[assignment] + rng.normal(0, cfg.cluster_sd_deg, n_clustered),
        rng.uniform(cfg.lat_min, cfg.lat_max, n_background),
    ])
    lon = np.concatenate([
        centres_lon[assignment] + rng.normal(0, cfg.cluster_sd_deg, n_clustered),
        rng.uniform(cfg.lon_min, cfg.lon_max, n_background),
    ])
    lat = np.clip(lat, cfg.lat_min, cfg.lat_max - 1e-9)
    lon = np.clip(lon, cfg.lon_min, cfg.lon_max - 1e-9)
    return lat, lon


def assign_spatial_blocks(lat: np.ndarray, lon: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    """Row-major integer id of the regular lat/lon grid cell containing each point."""
    n_cols = int(np.ceil((cfg.lon_max - cfg.lon_min) / cfg.block_size_deg))
    row = np.floor((lat - cfg.lat_min) / cfg.block_size_deg).astype(int)
    col = np.floor((lon - cfg.lon_min) / cfg.block_size_deg).astype(int)
    return row * n_cols + col


def generate_dataset(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    lat, lon = sample_locations(cfg, rng)
    n = lat.size

    # --- latent environmental fields -------------------------------------------------
    f_elev = _RandomField(rng, lengthscale_deg=0.50)(lat, lon)
    f_rain = _RandomField(rng, lengthscale_deg=0.40)(lat, lon)
    f_clay = _RandomField(rng, lengthscale_deg=0.30)(lat, lon)
    f_ph = _RandomField(rng, lengthscale_deg=0.35)(lat, lon)
    f_soc = _RandomField(rng, lengthscale_deg=0.15)(lat, lon)
    f_mgmt = _RandomField(rng, lengthscale_deg=0.08)(lat, lon)

    elevation = np.clip(1150 + 220 * f_elev + 120 * (lat - cfg.lat_min), 600, 2500)
    # Orographic rainfall: wetter uplands, plus an independent regional component.
    rainfall = np.clip(1150 + 0.45 * (elevation - 1150) + 200 * f_rain, 500, 2600)
    clay = np.clip(100 * _sigmoid(-0.4 + 0.9 * f_clay), 5, 75)
    # Leaching acidifies wetter soils.
    ph = np.clip(6.3 - 0.0014 * (rainfall - 1150) + 0.35 * f_ph, 4.2, 8.2)

    # --- SOC response (log scale) ----------------------------------------------------
    # Langmuir-type saturation: mineral-associated OC capacity grows with clay but
    # levels off, so the clay effect is concave rather than linear.
    clay_protection = clay / (clay + 25.0)
    moisture = np.tanh((rainfall - 1000.0) / 400.0)
    # Cooler uplands decompose litter more slowly.
    cooling = (elevation - 1150.0) / 1000.0
    # Microbial activity and nutrient availability peak near neutral pH.
    ph_penalty = (ph - 6.2) ** 2

    log_soc = (
        2.35
        + 1.10 * clay_protection
        + 0.45 * moisture
        + 0.40 * cooling
        - 0.10 * ph_penalty
        + 0.35 * clay_protection * moisture
        + 0.15 * f_soc
    )
    # Heteroscedastic process noise: wet sites mix waterlogged and drained profiles,
    # so SOC is more variable there. This is what the quantile model has to learn.
    process_sd = 0.10 + 0.14 * _sigmoid((rainfall - 1350.0) / 150.0)
    log_soc = log_soc + rng.normal(0.0, process_sd)
    soc_true = np.exp(log_soc)

    # --- Sentinel-2 surface reflectance via linear mixing -----------------------------
    # Vegetation fraction depends on water supply and local management, and only
    # weakly on SOC. The management field is spatially short-range and unrelated to
    # SOC, so it acts as a confounder in the optical signal.
    veg_fraction = _sigmoid(
        -0.2 + 1.6 * moisture + 0.9 * f_mgmt + 0.35 * (log_soc - 2.9)
    )
    veg_fraction = np.clip(veg_fraction, 0.02, 0.97)

    # Bare-soil brightness drops with SOC (organic matter darkens the surface).
    soil_red = 0.26 * np.exp(-0.018 * soc_true) + rng.normal(0, 0.01, n)
    soil_red = np.clip(soil_red, 0.05, 0.35)
    # Canopy chlorophyll varies with fertility; it shifts the red edge, which NDRE sees
    # and NDVI largely does not (NDVI saturates at high cover).
    chlorophyll = np.clip(0.5 + 0.25 * (log_soc - 2.9) + rng.normal(0, 0.1, n), 0.1, 1.0)

    v = veg_fraction
    blue = v * 0.030 + (1 - v) * 0.60 * soil_red   # B2, 490 nm
    red = v * 0.040 + (1 - v) * soil_red            # B4, 665 nm
    red_edge = v * (0.14 - 0.06 * chlorophyll) + (1 - v) * 1.10 * soil_red  # B5, 705 nm
    nir = v * 0.42 + (1 - v) * 1.30 * soil_red      # B8, 842 nm

    # Radiometric noise, roughly the L2A surface-reflectance uncertainty (~3 %).
    bands = [blue, red, red_edge, nir]
    blue, red, red_edge, nir = (b * (1 + rng.normal(0, 0.03, n)) for b in bands)

    # Haze / thin cloud that escaped masking adds a path-radiance term that is strongest
    # at short wavelengths. It compresses NDVI towards zero and biases SOC signal low.
    cloud_residual = rng.random(n) < cfg.cloud_residual_prob
    haze = np.where(cloud_residual, rng.uniform(0.02, 0.09, n), 0.0)
    blue = blue + 1.6 * haze
    red = red + 1.0 * haze
    red_edge = red_edge + 0.8 * haze
    nir = nir + 0.6 * haze

    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
    ndre = (nir - red_edge) / (nir + red_edge)

    # --- SoilGrids / climate-layer error ---------------------------------------------
    # SoilGrids values are themselves model predictions at 250 m, not measurements.
    # The error magnitudes are in the range of the published SoilGrids 2.0 accuracy
    # for topsoil pH and clay; elevation and rainfall errors mimic SRTM and gridded
    # climatologies.
    ph_obs = ph + rng.normal(0, 0.40, n)
    clay_obs = np.clip(clay + rng.normal(0, 7.0, n), 1, 90)
    elevation_obs = elevation + rng.normal(0, 6.0, n)
    rainfall_obs = rainfall * (1 + rng.normal(0, 0.06, n))

    # Dry combustion / Walkley-Black repeatability is a few percent of the value.
    soc_obs = soc_true * (1 + rng.normal(0, 0.05, n))

    df = pd.DataFrame({
        "sample_id": np.arange(n),
        "latitude": lat,
        "longitude": lon,
        "spatial_block_id": assign_spatial_blocks(lat, lon, cfg),
        "ndvi": ndvi,
        "evi": evi,
        "ndre": ndre,
        "ph_h2o": ph_obs,
        "clay_pct": clay_obs,
        "elevation_m": elevation_obs,
        "map_mm": rainfall_obs,
        TARGET_COLUMN: soc_obs,
        # Diagnostic only. In a real pipeline an undetected residual is by definition
        # unknown, so this column must never enter the feature matrix.
        "qa_cloud_residual": cloud_residual.astype(int),
    })
    return df


def to_geodataframe(df: pd.DataFrame, crs: str = "EPSG:4326"):
    """Wrap the table as a GeoDataFrame for export to GeoPackage or overlay in GIS."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs=crs
    )


def sample_raster_at_points(raster_path: str | Path, lat: np.ndarray, lon: np.ndarray,
                            band: int = 1) -> np.ndarray:
    """Read raster values at point locations, for swapping in real S2 / SoilGrids layers.

    Coordinates are reprojected from WGS84 into the raster CRS. Nodata pixels (e.g.
    cloud-masked S2 composites) are returned as NaN so they can be filtered before
    model fitting instead of silently becoming zeros.
    """
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(raster_path) as src:
        xs, ys = transform("EPSG:4326", src.crs, list(lon), list(lat))
        values = np.array([v[0] for v in src.sample(zip(xs, ys), indexes=band)], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=GeneratorConfig.n_samples)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "soc_synthetic.csv")
    args = parser.parse_args()

    df = generate_dataset(GeneratorConfig(n_samples=args.n_samples, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows, {df['spatial_block_id'].nunique()} spatial blocks -> {args.out}")


if __name__ == "__main__":
    main()
