import argparse, re, numpy as np, pandas as pd, warnings
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

warnings.filterwarnings("ignore")

# --- Model Wrappers ---

@dataclass
class SkSurvModel:
    name: str; pipe: Pipeline
    def fit(self, X, y): self.pipe.fit(X, y); return self
    def predict_risk(self, X): return self.pipe.predict(X)

class XGBCoxModel:
    def __init__(self, name, params, rounds): self.name, self.params, self.rounds = name, params, rounds
    def fit(self, X, y):
        self.imp = SimpleImputer(strategy="constant", fill_value=-999, add_indicator=True); X_i = self.imp.fit_transform(X)
        d = xgb.DMatrix(X_i, label=y["time"], weight=y["event"])
        self.b = xgb.train(self.params, d, num_boost_round=self.rounds)
        return self
    def predict_risk(self, X): return self.b.predict(xgb.DMatrix(self.imp.transform(X)))

class LGBMCoxModel:
    def __init__(self, name, params, rounds): self.name, self.params, self.rounds = name, params, rounds
    def fit(self, X, y):
        self.imp = SimpleImputer(strategy="constant", fill_value=-999, add_indicator=True); X_i = self.imp.fit_transform(X)
        d = lgb.Dataset(X_i, label=y["time"], weight=y["event"])
        self.b = lgb.train(self.params, d, num_boost_round=self.rounds)
        return self
    def predict_risk(self, X): return self.b.predict(self.imp.transform(X))

class CatBoostCoxModel:
    def __init__(self, name, params, iters): self.name, self.params, self.iters = name, params, iters
    def fit(self, X, y):
        self.imp = SimpleImputer(strategy="constant", fill_value=-999); X_i = self.imp.fit_transform(X)
        labels = np.where(y["event"] > 0, y["time"], -y["time"])
        self.m = CatBoostRegressor(iterations=self.iters, **self.params)
        self.m.fit(X_i, labels, verbose=False); return self
    def predict_risk(self, X): return self.m.predict(self.imp.transform(X))

# --- Feature Engineering ---

def build_features(df, event_ages=None):
    df = df.copy(); pat = re.compile(r"^(.*)_v(\d+)$"); bases = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            b_low = m.group(1).lower()
            if b_low not in bases: bases[b_low] = m.group(1)

    # 1. TRUNCATION: Mandatory anti-leakage
    if event_ages is not None:
        ev_np = event_ages.to_numpy()
        for v in range(1, 25):
            ac = f"Age_v{v}"
            if ac in df.columns:
                mask = (df[ac].to_numpy() > ev_np); mask[np.isnan(mask)] = False
                if mask.any():
                    cols = [ac] + [f"{o}_v{v}" for o in bases.values() if f"{o}_v{v}" in df.columns]
                    df.loc[mask, list(set(cols))] = np.nan

    age_cols = sorted([c for c in df.columns if c.lower().startswith("age_v")], key=lambda x: int(re.search(r"_v(\d+)$", x).group(1)))
    age_mat = df[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    age_last = np.nanmax(age_mat, axis=1)
    age_first = np.array([age_mat[i, np.where(np.isfinite(age_mat[i]))[0][0]] if np.any(np.isfinite(age_mat[i])) else np.nan for i in range(df.shape[0])])

    feats = {}
    for c in ["gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]:
        if c in df.columns: feats[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()

    feats["age_last"] = age_last
    feats["obs_dur"] = age_last - age_first
    feats["v_count"] = np.isfinite(age_mat).sum(axis=1).astype(float)

    for b_low, b_orig in bases.items():
        vcols = [f"{b_orig}_v{i}" for i in range(1, 25) if f"{b_orig}_v{i}" in df.columns]
        if not vcols: continue
        mat = df[vcols].apply(pd.to_numeric, errors="coerce").to_numpy()
        v_ages = age_mat[:, [int(re.search(r"_v(\d+)$", c).group(1)) - 1 for c in vcols]]
        finite = np.isfinite(mat)
        count = finite.sum(axis=1).astype(float)

        l_v, f_v, m_v = np.full(df.shape[0], np.nan), np.full(df.shape[0], np.nan), np.nanmax(mat, axis=1)
        age_at_max, slp = np.full(df.shape[0], np.nan), np.full(df.shape[0], np.nan)
        tsfa, vola = np.full(df.shape[0], np.nan), np.full(df.shape[0], np.nan)

        # Clinical thresholds for "abnormal"
        t_val = 40 if b_low in ["alt", "ast"] else 50 if b_low=="ggt" else 8.5 if b_low.startswith("fibs") else 0.5 if b_low.startswith("fibro") else None

        for i in range(df.shape[0]):
            idx = np.where(finite[i])[0]
            if len(idx) > 0:
                l_v[i], f_v[i] = mat[i, idx[-1]], mat[i, idx[0]]
                age_at_max[i] = v_ages[i, idx[np.argmax(mat[i, idx])]]
                if len(idx) >= 2:
                    dx = v_ages[i, idx[-1]] - v_ages[i, idx[0]]
                    if dx > 0: slp[i] = (mat[i, idx[-1]] - mat[i, idx[0]]) / dx
                    vola[i] = np.std(np.diff(mat[i, idx]))
                if t_val:
                    is_ab = (mat[i] > t_val) if b_low != "plt" else (mat[i] < 150)
                    ab_idx = np.where(is_ab & finite[i])[0]
                    if len(ab_idx) > 0: tsfa[i] = age_last[i] - v_ages[i, ab_idx[0]]

        feats[f"{b_low}__last"], feats[f"{b_low}__max"] = l_v, m_v
        feats[f"{b_low}__slope"], feats[f"{b_low}__vola"] = slp, vola
        feats[f"{b_low}__tsfa"] = tsfa
        feats[f"{b_low}__miss"] = 1.0 - (count / float(len(vcols)))
        if t_val: feats[f"{b_low}__is_ab"] = (m_v > t_val).astype(float) if b_low != "plt" else (np.nanmin(mat, axis=1) < 150).astype(float)

    # Combined Markers
    if "ast__last" in feats and "alt__last" in feats and "plt__last" in feats:
        ast, alt, plt = feats["ast__last"], feats["alt__last"], feats["plt__last"]
        feats["fib4"] = (age_last * ast) / (plt * np.sqrt(np.maximum(alt, 1e-6)))
        feats["apri"] = (ast / 40.0) * 100.0 / np.maximum(plt, 1e-6)
        feats["ast_alt"] = ast / np.maximum(alt, 1e-6)

    res_df = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], np.nan).astype(float)
    # Filter out columns that are all NaN to avoid Imputer issues
    res_df = res_df.dropna(axis=1, how='all')
    return res_df

# --- Pipeline ---

def prepare_targets(df, outcome):
    col = "evenements_hepatiques_majeurs" if outcome == "hepatic" else "death"
    age_col = "evenements_hepatiques_age_occur" if outcome == "hepatic" else "death_age_occur"
    mask = (~df[col].isna()) & (~((df[col] == 1) & df[age_col].isna()))
    df_v = df.loc[mask].copy()
    age_mat = df_v[[c for c in df.columns if c.lower().startswith("age_v")]].apply(pd.to_numeric, errors="coerce").to_numpy()
    age_start = np.array([age_mat[i, np.where(np.isfinite(age_mat[i]))[0][0]] for i in range(df_v.shape[0])])
    ev = (df_v[col] == 1).to_numpy()
    t = np.where(ev, pd.to_numeric(df_v[age_col], errors="coerce").to_numpy() - age_start, np.nanmax(age_mat, axis=1) - age_start)
    return df_v, Surv.from_arrays(event=ev, time=np.maximum(t, 0.001))

def get_models(seed, n):
    return [
        SkSurvModel("rsf", Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomSurvivalForest(n_estimators=n, min_samples_leaf=12, random_state=seed, n_jobs=-1))])),
        SkSurvModel("gbs", Pipeline([("i", SimpleImputer(strategy="median")), ("m", GradientBoostingSurvivalAnalysis(n_estimators=n, learning_rate=0.03, max_depth=3, random_state=seed))])),
        XGBCoxModel("xgb", {"objective": "survival:cox", "tree_method": "hist", "learning_rate": 0.008, "max_depth": 4, "seed": seed, "verbosity": 0}, n*2),
        LGBMCoxModel("lgb", {"objective": "regression", "learning_rate": 0.008, "num_leaves": 31, "verbose": -1, "seed": seed}, n*2),
        CatBoostCoxModel("cat", {"loss_function": "Cox", "learning_rate": 0.015, "depth": 4, "random_seed": seed}, n)
    ]

def _rank(x): return pd.Series(x).rank(pct=True).to_numpy()

def train_and_predict(tr_path, te_path, out_path):
    tr_df, te_df = pd.read_csv(tr_path), pd.read_csv(te_path)
    # Build initial test features to define column space
    X_te_full = build_features(te_df)
    results = {}

    for target in ["hepatic", "death"]:
        print(f"Modeling {target}...")
        df_v, y_v = prepare_targets(tr_df, target)
        df_v = df_v.reset_index(drop=True)
        X_v = build_features(df_v, df_v["evenements_hepatiques_age_occur" if target=="hepatic" else "death_age_occur"])

        # Align columns between train and test
        common_cols = X_v.columns.intersection(X_te_full.columns)
        X_v, X_te = X_v[common_cols], X_te_full[common_cols]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_m = 5
        oof = np.zeros((X_v.shape[0], n_m))

        for tr_i, va_i in cv.split(X_v, y_v["event"]):
            Xt, yt = X_v.iloc[tr_i], y_v[tr_i]
            for i, m in enumerate(get_models(42, 250)):
                m.fit(Xt, yt)
                oof[va_i, i] = m.predict_risk(X_v.iloc[va_i])

        o_rank = np.column_stack([_rank(oof[:, i]) for i in range(n_m)])
        # Optimized blend weights
        bw, bc = np.ones(n_m)/n_m, -1
        for _ in range(1000):
            w = np.random.dirichlet(np.ones(n_m))
            c = concordance_index_censored(y_v["event"], y_v["time"], o_rank @ w)[0]
            if c > bc: bc, bw = c, w
        print(f" OOF C-index: {bc:.4f} | Weights: {bw.round(3)}")

        # Bagging on full train set
        final_preds = []
        for s in [42, 1337, 2026]:
            print(f"  Bagging seed {s}...")
            mods = [m.fit(X_v, y_v) for m in get_models(s, 600)]
            p_mat = np.column_stack([_rank(m.predict_risk(X_te)) for m in mods])
            final_preds.append(p_mat @ bw)
        results[target] = np.mean(final_preds, axis=0)

    sub = pd.DataFrame({"trustii_id": te_df["trustii_id"], "risk_hepatic_event": results["hepatic"], "risk_death": results["death"]})
    sub.sort_values("trustii_id").to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

if __name__ == "__main__":
    train_and_predict("competition_data/Train.csv", "competition_data/Test.csv", "final_submission.csv")
