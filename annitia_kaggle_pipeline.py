import argparse
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

import xgboost as xgb

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis


TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

ID_COLS_TRAIN = ["patient_id_anon"]
ID_COLS_TEST = ["trustii_id", "patient_id_anon"]


def _visit_bases_and_max_visits(columns: list[str]) -> tuple[list[str], dict[str, int]]:
    pat = re.compile(r"^(.*)_v(\d+)$")
    bases: set[str] = set()
    maxv: dict[str, int] = {}
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        b = m.group(1)
        v = int(m.group(2))
        bases.add(b)
        maxv[b] = max(maxv.get(b, 0), v)
    return sorted(bases), maxv


def _rowwise_first(values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    idx = np.where(mask, np.arange(values.shape[1])[None, :], values.shape[1])
    first_idx = idx.min(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    valid = first_idx < values.shape[1]
    out[valid] = values[np.arange(values.shape[0])[valid], first_idx[valid]].astype(float)
    return out


def _rowwise_last(values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    idx = np.where(mask, np.arange(values.shape[1])[None, :], -1)
    last_idx = idx.max(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    valid = last_idx >= 0
    out[valid] = values[np.arange(values.shape[0])[valid], last_idx[valid]].astype(float)
    return out


def _rowwise_last_k_mean(values: np.ndarray, k: int) -> np.ndarray:
    n, t = values.shape
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        y = values[i]
        m = np.isfinite(y)
        if not m.any():
            continue
        idx = np.where(m)[0]
        tail = idx[-k:]
        out[i] = float(np.nanmean(y[tail]))
    return out


def _slope(values: np.ndarray) -> np.ndarray:
    n, t = values.shape
    x = np.arange(t, dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        y = values[i]
        m = np.isfinite(y)
        if m.sum() < 2:
            continue
        xi = x[m]
        yi = y[m]
        xm = xi.mean()
        ym = yi.mean()
        denom = ((xi - xm) ** 2).sum()
        if denom <= 0:
            continue
        out[i] = ((xi - xm) * (yi - ym)).sum() / denom
    return out


def _ewm_last(values: np.ndarray, alpha: float) -> np.ndarray:
    n, t = values.shape
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        y = values[i]
        m = np.isfinite(y)
        if not m.any():
            continue
        yi = y[m]
        e = float(yi[0])
        for v in yi[1:]:
            e = alpha * float(v) + (1.0 - alpha) * e
        out[i] = e
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    bases, maxv = _visit_bases_and_max_visits(df.columns.tolist())

    static_cols = [
        c
        for c in df.columns
        if (c not in TARGET_COLS)
        and not re.match(r"^(.*)_v\d+$", c)
        and c not in ID_COLS_TRAIN
        and c not in ID_COLS_TEST
    ]

    feats: dict[str, np.ndarray] = {}

    for c in static_cols:
        feats[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

    age_cols = [c for c in df.columns if c.startswith("Age_v")]
    if age_cols:
        age_mat = df[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        age_baseline = age_mat[:, 0]
        age_last = np.nanmax(age_mat, axis=1)
        feats["age__baseline"] = age_baseline
        feats["age__last"] = age_last
        feats["age__followup"] = age_last - age_baseline
        feats["age__n_visits"] = np.isfinite(age_mat).sum(axis=1).astype(float)

    for base in bases:
        vcols = [f"{base}_v{i}" for i in range(1, maxv[base] + 1) if f"{base}_v{i}" in df.columns]
        if not vcols:
            continue

        mat = df[vcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(mat)
        count = finite.sum(axis=1).astype(float)
        frac_missing = 1.0 - (count / float(mat.shape[1]))

        first = _rowwise_first(mat)
        last = _rowwise_last(mat)
        last3 = _rowwise_last_k_mean(mat, 3)
        last5 = _rowwise_last_k_mean(mat, 5)

        with np.errstate(all="ignore"):
            mean = np.nanmean(mat, axis=1)
            std = np.nanstd(mat, axis=1)
            q10 = np.nanquantile(mat, 0.10, axis=1)
            q50 = np.nanquantile(mat, 0.50, axis=1)
            q90 = np.nanquantile(mat, 0.90, axis=1)
            vmin = np.nanmin(mat, axis=1)
            vmax = np.nanmax(mat, axis=1)

        all_nan = count == 0
        if np.any(all_nan):
            mean = np.where(all_nan, np.nan, mean)
            std = np.where(all_nan, np.nan, std)
            q10 = np.where(all_nan, np.nan, q10)
            q50 = np.where(all_nan, np.nan, q50)
            q90 = np.where(all_nan, np.nan, q90)
            vmin = np.where(all_nan, np.nan, vmin)
            vmax = np.where(all_nan, np.nan, vmax)

        slope = _slope(mat)
        ewm = _ewm_last(mat, alpha=0.4)

        delta = last - first
        range_ = vmax - vmin

        # Recent-vs-history signals (often strong for progression)
        delta_last_mean = last - mean
        delta_last3_mean = last3 - mean
        delta_last_ewm = last - ewm

        feats[f"{base}__count"] = count
        feats[f"{base}__frac_missing"] = frac_missing
        feats[f"{base}__first"] = first
        feats[f"{base}__last"] = last
        feats[f"{base}__last3"] = last3
        feats[f"{base}__last5"] = last5
        feats[f"{base}__delta"] = delta
        feats[f"{base}__mean"] = mean
        feats[f"{base}__std"] = std
        feats[f"{base}__q10"] = q10
        feats[f"{base}__q50"] = q50
        feats[f"{base}__q90"] = q90
        feats[f"{base}__min"] = vmin
        feats[f"{base}__max"] = vmax
        feats[f"{base}__range"] = range_
        feats[f"{base}__slope"] = slope
        feats[f"{base}__ewm"] = ewm
        feats[f"{base}__last_minus_mean"] = delta_last_mean
        feats[f"{base}__last3_minus_mean"] = delta_last3_mean
        feats[f"{base}__last_minus_ewm"] = delta_last_ewm

        if base.lower() in {"ast", "alt", "ggt", "bilirubin", "triglyc", "chol", "gluc_fast"}:
            feats[f"{base}__last_log1p"] = np.log1p(np.maximum(last, 0.0))
            feats[f"{base}__ewm_log1p"] = np.log1p(np.maximum(ewm, 0.0))

    if age_cols:
        if "ast_v1" in df.columns and "plt_v1" in df.columns:
            ast_last = feats.get("ast__last", pd.to_numeric(df.get("ast_v1"), errors="coerce").to_numpy(dtype=float))
            plt_last = feats.get("plt__last", pd.to_numeric(df.get("plt_v1"), errors="coerce").to_numpy(dtype=float))
            age_last = feats.get("age__last", pd.to_numeric(df[age_cols[-1]], errors="coerce").to_numpy(dtype=float))

            alt_last = feats.get("alt__last", np.nan)
            if isinstance(alt_last, float):
                alt_last = pd.to_numeric(df.get("alt_v1"), errors="coerce").to_numpy(dtype=float)

            fib4 = (age_last * ast_last) / (plt_last * np.sqrt(np.maximum(alt_last, 1e-6)))
            feats["fib4__last"] = fib4

            apri = (ast_last / 40.0) * 100.0 / np.maximum(plt_last, 1e-6)
            feats["apri__last"] = apri

            # Platelet-to-ALT ratio (inverse of liver stress)
            feats["plt_alt_ratio__last"] = plt_last / np.maximum(alt_last, 1e-6)

            # AST/ALT ratio (common in hepatology)
            feats["ast_alt_ratio__last"] = ast_last / np.maximum(alt_last, 1e-6)

    X = pd.DataFrame(feats, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=["number"]).astype(float)
    return X


def prepare_survival_targets(train_df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, np.ndarray]:
    if outcome == "hepatic":
        event_col = "evenements_hepatiques_majeurs"
        age_occur_col = "evenements_hepatiques_age_occur"
        name_event = "Hepatic_event"
        unknown_mask = pd.Series(False, index=train_df.index)
    elif outcome == "death":
        event_col = "death"
        age_occur_col = "death_age_occur"
        name_event = "Death"
        unknown_mask = train_df[event_col].isna()
    else:
        raise ValueError("outcome must be one of: hepatic, death")

    age_cols = [c for c in train_df.columns if c.startswith("Age_v")]
    age_mat = train_df[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    baseline_age = age_mat[:, 0]
    last_age = np.nanmax(age_mat, axis=1)

    is_event = train_df[event_col] == 1
    invalid = is_event & train_df[age_occur_col].isna()
    mask = (~unknown_mask) & (~invalid)

    df_valid = train_df.loc[mask].copy().reset_index(drop=True)

    age_cols_v = [c for c in df_valid.columns if c.startswith("Age_v")]
    age_mat_v = df_valid[age_cols_v].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    baseline_age_v = age_mat_v[:, 0]
    last_age_v = np.nanmax(age_mat_v, axis=1)

    is_event_v = (df_valid[event_col] == 1).astype(bool)
    t = np.where(
        is_event_v,
        pd.to_numeric(df_valid[age_occur_col], errors="coerce").to_numpy(dtype=float) - baseline_age_v,
        last_age_v - baseline_age_v,
    ).astype(float)
    t = np.maximum(t, 0.001)

    y = Surv.from_arrays(event=is_event_v.to_numpy(), time=t, name_event=name_event, name_time="Time_years")
    return df_valid, y


@dataclass
class SkSurvModel:
    name: str
    pipe: Pipeline

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SkSurvModel":
        self.pipe.fit(X, y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.pipe.predict(X)
        return np.asarray(pred, dtype=float)


@dataclass
class XGBCoxModel:
    name: str
    params: dict
    num_boost_round: int
    booster: xgb.Booster | None = None
    imputer: SimpleImputer | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBCoxModel":
        if y.dtype.names is None or len(y.dtype.names) < 2:
            raise ValueError("Survival target y must be a structured array with (event, time) fields")
        event_field, time_field = y.dtype.names[0], y.dtype.names[1]

        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X)
        times = np.asarray(y[time_field], dtype=float)
        weights = np.asarray(y[event_field], dtype=float)

        dtrain = xgb.DMatrix(X_imp, label=times, weight=weights, missing=np.nan)
        try:
            self.booster = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        except xgb.core.XGBoostError:
            # Some Kaggle images ship CPU-only XGBoost even when a GPU is attached.
            # Fall back to CPU-safe params.
            cpu_params = dict(self.params)
            cpu_params.pop("device", None)
            cpu_params["tree_method"] = "hist"
            self.booster = xgb.train(cpu_params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None or self.imputer is None:
            raise RuntimeError("XGBCoxModel must be fit() before predict_risk().")
        X_imp = self.imputer.transform(X)
        dtest = xgb.DMatrix(X_imp, missing=np.nan)
        return np.asarray(self.booster.predict(dtest), dtype=float)


def make_models(seed: int) -> list[SkSurvModel]:
    rsf = SkSurvModel(
        name="rsf",
        pipe=Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                (
                    "m",
                    RandomSurvivalForest(
                        n_estimators=800,
                        min_samples_split=30,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    )

    gb = SkSurvModel(
        name="gbsa",
        pipe=Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                (
                    "m",
                    GradientBoostingSurvivalAnalysis(
                        loss="coxph",
                        learning_rate=0.03,
                        n_estimators=800,
                        max_depth=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    )

    coxnet = SkSurvModel(
        name="coxnet",
        pipe=Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
                (
                    "m",
                    CoxnetSurvivalAnalysis(
                        l1_ratio=0.2,
                        alpha_min_ratio=0.01,
                        n_alphas=60,
                        max_iter=100000,
                    ),
                ),
            ]
        ),
    )

    # XGBoost Cox: strong on tabular. Try CUDA (if XGBoost was built with GPU support),
    # but fall back automatically in XGBCoxModel.fit().
    xgb_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 3,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": int(seed),
    }

    xgb_cox = XGBCoxModel(name="xgb_cox", params=xgb_params, num_boost_round=2500)

    return [rsf, gb, coxnet, xgb_cox]


def make_models_by_mode(seed: int, mode: str) -> list:
    mode = str(mode).lower().strip()
    all_models = make_models(seed)
    if mode == "baseline":
        return [m for m in all_models if getattr(m, "name", "") in {"rsf", "gbsa", "coxnet"}]
    if mode == "xgb":
        return [m for m in all_models if getattr(m, "name", "") == "xgb_cox"]
    if mode == "blend":
        return all_models
    raise ValueError("mode must be one of: baseline, xgb, blend")


def _rank_transform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    ranks /= max(len(x) - 1, 1)
    return ranks


def _make_cv_splitter(y: np.ndarray, n_splits: int, seed: int):
    # Stratify by event indicator when possible; this stabilizes C-index estimates.
    if y.dtype.names is None or len(y.dtype.names) < 1:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    event_field = y.dtype.names[0]
    labels = np.asarray(y[event_field], dtype=int)
    if np.unique(labels).size < 2:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _simplex_grid_weights(n_models: int, step: float = 0.05) -> np.ndarray:
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    if n_models == 1:
        return np.ones((1, 1), dtype=float)

    grid = np.arange(0.0, 1.0 + 1e-12, step, dtype=float)

    weights: list[list[float]] = []
    if n_models == 2:
        for a in grid:
            weights.append([a, 1.0 - a])
        return np.asarray(weights, dtype=float)

    if n_models == 3:
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c < -1e-12:
                    continue
                if c < 0:
                    c = 0.0
                weights.append([a, b, c])
        return np.asarray(weights, dtype=float)

    # n_models >= 4: do random Dirichlet (grid explodes)
    rng = np.random.default_rng(123)
    return rng.dirichlet(np.ones(n_models, dtype=float), size=4000).astype(float)


def _weight_search(
    y: np.ndarray,
    oof_rank_preds: np.ndarray,
    seed: int,
) -> np.ndarray:
    if y.dtype.names is None or len(y.dtype.names) < 2:
        raise ValueError("Survival target y must be a structured array with (event, time) fields")
    event_field, time_field = y.dtype.names[0], y.dtype.names[1]

    n_models = oof_rank_preds.shape[1]
    best_w = np.ones(n_models, dtype=float) / float(n_models)
    best_c = -1.0

    # Always evaluate equal weights
    base = oof_rank_preds @ best_w
    best_c = float(concordance_index_censored(y[event_field], y[time_field], base)[0])

    candidates = _simplex_grid_weights(n_models=n_models, step=0.05)
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    for w in candidates:
        w = np.asarray(w, dtype=float)
        s = float(w.sum())
        if s <= 0:
            continue
        w = w / s
        pred = oof_rank_preds @ w
        c = float(concordance_index_censored(y[event_field], y[time_field], pred)[0])
        if c > best_c:
            best_c = c
            best_w = w
    return best_w


def cv_fit_and_blend(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    n_splits: int = 4,
    mode: str = "blend",
) -> tuple[list, np.ndarray]:
    cv = _make_cv_splitter(y=y, n_splits=n_splits, seed=seed)
    models = make_models_by_mode(seed, mode)

    n = X.shape[0]
    k = len(models)
    oof = np.full((n, k), np.nan, dtype=float)

    if isinstance(cv, StratifiedKFold):
        splits = cv.split(X, np.asarray(y[y.dtype.names[0]], dtype=int))
    else:
        splits = cv.split(X)

    for tr, va in splits:
        X_tr = X.iloc[tr]
        X_va = X.iloc[va]
        y_tr = y[tr]

        fold_models = make_models_by_mode(seed, mode)
        for mi, m in enumerate(fold_models):
            m.fit(X_tr, y_tr)
            oof[va, mi] = m.predict_risk(X_va)

    # Rank-transform each model's OOF predictions (robust to scale differences)
    oof_rank = np.vstack([_rank_transform(oof[:, i]) for i in range(k)]).T

    weights = _weight_search(y=y, oof_rank_preds=oof_rank, seed=seed + 999)

    # Fit final models on full data
    fitted = make_models_by_mode(seed, mode)
    for m in fitted:
        m.fit(X, y)

    return fitted, weights


def ensemble_predict(models: list, X: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    preds = []
    for m in models:
        preds.append(_rank_transform(m.predict_risk(X)))
    mat = np.vstack(preds).T
    w = np.asarray(weights, dtype=float)
    w = w / np.maximum(w.sum(), 1e-12)
    return mat @ w


def fit_and_predict(train_path: str, test_path: str, output_path: str) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    df_hep, y_hep = prepare_survival_targets(train_df, outcome="hepatic")
    df_death, y_death = prepare_survival_targets(train_df, outcome="death")

    X_hep = build_features(df_hep)
    X_death = build_features(df_death)
    X_test = build_features(test_df)

    # mode controls model set:
    # - baseline: rsf + gbsa + coxnet
    # - xgb: xgboost cox only
    # - blend: baseline + xgb
    mode = getattr(fit_and_predict, "mode", "blend")

    n_splits = int(getattr(fit_and_predict, "n_splits", 4))
    bag_seeds = int(getattr(fit_and_predict, "bag_seeds", 1))
    seed0 = int(getattr(fit_and_predict, "seed0", 42))

    def _bagged_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, seed_base: int) -> np.ndarray:
        preds = []
        for i in range(max(bag_seeds, 1)):
            s = seed_base + 1000 * i
            models, w = cv_fit_and_blend(X_tr, y_tr, seed=s, n_splits=n_splits, mode=mode)
            preds.append(_rank_transform(ensemble_predict(models, X_te, w)))
        return np.mean(np.vstack(preds), axis=0)

    risk_hep = _bagged_predict(X_hep, y_hep, X_test, seed_base=seed0)
    risk_death = _bagged_predict(X_death, y_death, X_test, seed_base=seed0 + 100)

    if "trustii_id" not in test_df.columns:
        raise ValueError("Test file must contain trustii_id")

    sub = pd.DataFrame(
        {
            "trustii_id": test_df["trustii_id"].astype(int).to_numpy(),
            "risk_hepatic_event": risk_hep.astype(float),
            "risk_death": risk_death.astype(float),
        }
    ).sort_values("trustii_id")

    sub.to_csv(output_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="blend", choices=["baseline", "xgb", "blend"])
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--bag_seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fit_and_predict.mode = args.mode
    fit_and_predict.n_splits = args.n_splits
    fit_and_predict.bag_seeds = args.bag_seeds
    fit_and_predict.seed0 = args.seed
    fit_and_predict(args.train, args.test, args.out)


if __name__ == "__main__":
    main()
