"""Quantile gradient boosting for SOC with spatial block cross-validation.

One LightGBM model is fitted per quantile tau in {0.05, 0.50, 0.95} by minimising the
pinball loss

    L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q)),

whose population minimiser is the conditional tau-quantile. [Q_0.05, Q_0.95] is then a
nominal 90 % prediction interval. Nominal is not the same as achieved: separately
fitted quantile models can cross, and on spatially held-out data they tend to
under-cover. Both issues are handled explicitly below.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import BallTree

QUANTILES = (0.05, 0.50, 0.95)
EARTH_RADIUS_KM = 6371.0088


def set_global_seed(seed: int = 42) -> None:
    """Fix every RNG the pipeline can touch.

    scikit-learn and LightGBM take their seeds per estimator (random_state / seed);
    those are passed explicitly below. torch is optional and only seeded if present,
    so the fusion network from the imaging project can share this helper.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass
class QuantileGBMConfig:
    n_estimators: int = 600
    learning_rate: float = 0.03
    # Shallow trees: with ~1.5k samples and 7 features, deeper trees fit the
    # tails of the quantile loss to a handful of points and the intervals get noisy.
    num_leaves: int = 15
    min_child_samples: int = 25
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 42

    def lgb_params(self, alpha: float) -> dict:
        return dict(
            objective="quantile",
            alpha=alpha,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,  # multithreaded histogram building is not bit-reproducible
            verbose=-1,
        )


@dataclass
class QuantileSOCModel:
    """Set of per-quantile LightGBM regressors on a log-transformed target.

    Fitting on log(SOC) is exact for quantiles: for a strictly increasing g,
    Q_tau(g(Y)) = g(Q_tau(Y)), so exponentiating the log-scale quantiles returns SOC
    quantiles with no retransformation bias (unlike the mean). The log scale also
    turns the multiplicative noise in SOC into roughly additive noise, which trees
    handle better.
    """

    quantiles: tuple[float, ...] = QUANTILES
    config: QuantileGBMConfig = field(default_factory=QuantileGBMConfig)
    log_target: bool = True
    models: dict = field(default_factory=dict, init=False)
    # Additive CQR correction on the working (log) scale; 0 means uncalibrated.
    conformal_offset: float = field(default=0.0, init=False)

    def _to_working(self, y):
        return np.log(y) if self.log_target else np.asarray(y, dtype=float)

    def _from_working(self, z):
        return np.exp(z) if self.log_target else z

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuantileSOCModel":
        z = self._to_working(y)
        self.models = {}
        for tau in self.quantiles:
            m = lgb.LGBMRegressor(**self.config.lgb_params(alpha=tau))
            m.fit(X, z)
            self.models[tau] = m
        return self

    def _predict_working(self, X: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([self.models[tau].predict(X) for tau in self.quantiles])
        # Independently fitted quantile models can cross. Sorting each row is the
        # rearrangement of Chernozhukov et al. (2010): it never increases pinball loss
        # and guarantees Q_0.05 <= Q_0.50 <= Q_0.95.
        z = np.sort(z, axis=1)
        z[:, 0] -= self.conformal_offset
        z[:, -1] += self.conformal_offset
        return z

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, n_quantiles) array of SOC quantiles in g/kg."""
        return self._from_working(self._predict_working(X))

    def conformalize(self, X_cal: pd.DataFrame, y_cal: np.ndarray, alpha: float = 0.10) -> float:
        """Conformalized quantile regression (Romano et al., 2019).

        Nonconformity score E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)); the interval is
        widened (or narrowed, if E is mostly negative) by the ceil((n+1)(1-alpha))/n
        empirical quantile of E. The finite-sample guarantee assumes calibration and
        test points are exchangeable. Under spatial shift they are not, which is why the
        calibration set is made of held-out *blocks* rather than random points.
        """
        self.conformal_offset = 0.0
        z = self._to_working(y_cal)
        q = self._predict_working(X_cal)
        scores = np.maximum(q[:, 0] - z, z - q[:, -1])
        n = scores.size
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.conformal_offset = float(np.quantile(scores, level, method="higher"))
        return self.conformal_offset


# ---------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------

def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def interval_metrics(y: np.ndarray, lower: np.ndarray, median: np.ndarray, upper: np.ndarray,
                     alpha: float = 0.10) -> dict:
    """Point accuracy of the median and calibration/sharpness of the interval.

    PICP  : empirical coverage probability, share of y inside [lower, upper].
    MPIW  : mean prediction interval width (g/kg).
    NMPIW : MPIW divided by the observed range of y, for comparing across datasets.
    Interval score (Gneiting & Raftery, 2007): width plus 2/alpha times the miss
    distance. It rewards sharp intervals but penalises misses, so a model cannot
    lower it simply by narrowing intervals at the cost of coverage.
    """
    y = np.asarray(y, dtype=float)
    inside = (y >= lower) & (y <= upper)
    width = upper - lower
    interval_score = width + (2 / alpha) * ((lower - y) * (y < lower) + (y - upper) * (y > upper))
    return {
        "n": int(y.size),
        "r2": r2_score(y, median),
        "mae": mean_absolute_error(y, median),
        "rmse": float(np.sqrt(mean_squared_error(y, median))),
        "picp": float(inside.mean()),
        "mpiw": float(width.mean()),
        "nmpiw": float(width.mean() / (y.max() - y.min())),
        "interval_score": float(interval_score.mean()),
        "pinball_q05": pinball_loss(y, lower, 0.05),
        "pinball_q50": pinball_loss(y, median, 0.50),
        "pinball_q95": pinball_loss(y, upper, 0.95),
    }


# ---------------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------------

def spatial_buffer_mask(train_latlon: np.ndarray, test_latlon: np.ndarray, buffer_km: float) -> np.ndarray:
    """True for training points farther than buffer_km from every test point.

    Block CV alone still leaves training points a few metres across the block edge
    from test points. With spatially autocorrelated residuals those neighbours leak
    information and inflate test scores. Dropping a buffer ring removes that leak at
    the cost of some training data.
    """
    if buffer_km <= 0:
        return np.ones(len(train_latlon), dtype=bool)
    tree = BallTree(np.radians(test_latlon), metric="haversine")
    dist, _ = tree.query(np.radians(train_latlon), k=1)
    return dist[:, 0] * EARTH_RADIUS_KM > buffer_km


def make_spatial_folds(groups: np.ndarray, n_splits: int = 5):
    """GroupKFold over spatial blocks: every block is in exactly one test fold.

    GroupKFold balances fold sizes by sample count, so a heavily sampled block is not
    split between train and test. Random KFold on these data would put near-duplicate
    neighbours on both sides and report optimistic accuracy.
    """
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def cross_validate_spatial(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    group_col: str = "spatial_block_id",
    n_splits: int = 5,
    buffer_km: float = 0.0,
    calibration_fraction: float = 0.2,
    alpha: float = 0.10,
    config: QuantileGBMConfig | None = None,
    folds: list | None = None,
) -> pd.DataFrame:
    """Out-of-fold quantile predictions, raw and conformalized.

    Per outer fold two model sets are fitted:
      * raw  - all (buffered) training blocks, pinball loss only;
      * cqr  - a subset of training blocks, then conformalized on the remaining
               calibration blocks. Calibration blocks are never used for fitting.
    Holding out whole blocks for calibration mimics the test-time situation (new,
    unsampled areas); calibrating on random points would under-estimate the offset.
    """
    config = config or QuantileGBMConfig()
    groups = df[group_col].to_numpy()
    latlon = df[["latitude", "longitude"]].to_numpy()
    X = df[features]
    y = df[target].to_numpy()
    folds = folds if folds is not None else make_spatial_folds(groups, n_splits)

    records = []
    for k, (train_idx, test_idx) in enumerate(folds):
        keep = spatial_buffer_mask(latlon[train_idx], latlon[test_idx], buffer_km)
        train_idx = train_idx[keep]

        raw = QuantileSOCModel(config=config).fit(X.iloc[train_idx], y[train_idx])
        q_raw = raw.predict(X.iloc[test_idx])

        splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_fraction,
                                     random_state=config.seed + k)
        fit_rel, cal_rel = next(splitter.split(train_idx, groups=groups[train_idx]))
        fit_idx, cal_idx = train_idx[fit_rel], train_idx[cal_rel]
        cqr = QuantileSOCModel(config=config).fit(X.iloc[fit_idx], y[fit_idx])
        offset = cqr.conformalize(X.iloc[cal_idx], y[cal_idx], alpha=alpha)
        q_cqr = cqr.predict(X.iloc[test_idx])

        records.append(pd.DataFrame({
            "row": test_idx,
            "fold": k,
            "n_train": len(train_idx),
            "n_dropped_buffer": int((~keep).sum()),
            "cqr_offset_log": offset,
            "q05": q_raw[:, 0], "q50": q_raw[:, 1], "q95": q_raw[:, 2],
            "q05_cqr": q_cqr[:, 0], "q50_cqr": q_cqr[:, 1], "q95_cqr": q_cqr[:, 2],
        }))

    oof = pd.concat(records).set_index("row").sort_index()
    return df.join(oof)


def summarise_cv(oof: pd.DataFrame, target: str, alpha: float = 0.10) -> pd.DataFrame:
    """Per-fold and pooled metrics for the raw and conformalized intervals."""
    rows = []
    for variant, suffix in (("raw", ""), ("cqr", "_cqr")):
        cols = [f"q05{suffix}", f"q50{suffix}", f"q95{suffix}"]
        for fold, part in oof.groupby("fold"):
            m = interval_metrics(part[target], *(part[c] for c in cols), alpha=alpha)
            rows.append({"variant": variant, "fold": str(fold), **m})
        m = interval_metrics(oof[target], *(oof[c] for c in cols), alpha=alpha)
        rows.append({"variant": variant, "fold": "pooled", **m})
    return pd.DataFrame(rows)


def main() -> None:
    from pathlib import Path

    from data_generator import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset

    set_global_seed(42)
    csv = Path(__file__).resolve().parents[1] / "data" / "soc_synthetic.csv"
    df = pd.read_csv(csv) if csv.exists() else generate_dataset()
    oof = cross_validate_spatial(df, FEATURE_COLUMNS, TARGET_COLUMN, buffer_km=5.0)
    summary = summarise_cv(oof, TARGET_COLUMN)
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(summary[summary["fold"] == "pooled"].drop(columns="fold").to_string(index=False))


if __name__ == "__main__":
    main()
