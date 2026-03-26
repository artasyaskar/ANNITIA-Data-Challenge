"""
ANNITIA Challenge: Risk Stratification of MASLD through AI.
Predicting Liver Events (70%) and Death (30%) using Longitudinal NITs.
Developed for maximum accuracy and code quality.
"""

import argparse
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Model Wrappers for High Readability and Correctness ---

@dataclass
class SkSurvWrapper:
    """Wrapper for scikit-survival models to ensure consistent API."""
    name: str
    pipe: Pipeline

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict_risk(self, X):
        return self.pipe.predict(X)

class XGBoostSurvivalModel:
    """Correct XGBoost implementation for Cox Proportional Hazards."""
    def __init__(self, name, params, rounds):
        self.name = name
        self.params = params
        self.rounds = rounds
        self.imputer = SimpleImputer(strategy="constant", fill_value=-999, add_indicator=True)

    def fit(self, X, y):
        # XGBoost Cox: label is time, weight is event indicator
        X_imputed = self.imputer.fit_transform(X)
        dtrain = xgb.DMatrix(X_imputed, label=y["time"], weight=y["event"])
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.rounds)
        return self

    def predict_risk(self, X):
        X_imputed = self.imputer.transform(X)
        dtest = xgb.DMatrix(X_imputed)
        return self.booster.predict(dtest)

class LightGBMSurvivalProxy:
    """LightGBM proxy for survival ranking using event-weighted regression."""
    def __init__(self, name, params, rounds):
        self.name = name
        self.params = params
        self.rounds = rounds
        self.imputer = SimpleImputer(strategy="constant", fill_value=-999, add_indicator=True)

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        # Higher risk = shorter survival time. Regress on -time for event cases.
        # This acts as a ranking proxy for C-index.
        dtrain = lgb.Dataset(X_imputed, label=-y["time"], weight=y["event"])
        self.booster = lgb.train(self.params, dtrain, num_boost_round=self.rounds)
        return self

    def predict_risk(self, X):
        X_imputed = self.imputer.transform(X)
        return self.booster.predict(X_imputed)

class CatBoostSurvivalModel:
    """CatBoost implementation for Cox Proportional Hazards."""
    def __init__(self, name, params, iters):
        self.name = name
        self.params = params
        self.iters = iters
        self.imputer = SimpleImputer(strategy="constant", fill_value=-999)

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        # CatBoost Cox: Label > 0 for event, Label < 0 for censored
        labels = np.where(y["event"] > 0, y["time"], -y["time"])
        self.model = CatBoostRegressor(iterations=self.iters, **self.params)
        self.model.fit(X_imputed, labels, verbose=False)
        return self

    def predict_risk(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed)

# --- Enhanced Feature Engineering ---

def build_features(df, event_ages=None):
    """
    Extracts longitudinal, clinical, and temporal features.
    Implements anti-leakage truncation if event_ages is provided.
    """
    df = df.copy()

    # Identify longitudinal biomarkers and their visit suffixes (_v1, _v2, etc.)
    visit_pattern = re.compile(r"^(.*)_v(\d+)$")
    biomarker_bases = {}
    for col in df.columns:
        match = visit_pattern.match(col)
        if match:
            base_name = match.group(1).lower()
            if base_name not in biomarker_bases:
                biomarker_bases[base_name] = match.group(1)

    # 1. ANTI-LEAKAGE TRUNCATION
    # Remove observations that occur AFTER the event age to prevent looking into the future.
    if event_ages is not None:
        event_ages_np = event_ages.to_numpy()
        for visit_num in range(1, 25):
            age_col = f"Age_v{visit_num}"
            if age_col in df.columns:
                # Mask patients where this visit's age > their event age
                future_mask = (df[age_col].to_numpy() > event_ages_np)
                future_mask[np.isnan(future_mask)] = False
                if future_mask.any():
                    # Nullify the age and all associated biomarker measurements for this visit
                    visit_cols = [age_col] + [f"{b}_v{visit_num}" for b in biomarker_bases.values() if f"{b}_v{visit_num}" in df.columns]
                    df.loc[future_mask, list(set(visit_cols))] = np.nan

    # Pre-calculate age matrices
    age_cols = sorted([c for c in df.columns if c.lower().startswith("age_v")],
                      key=lambda x: int(re.search(r"_v(\d+)$", x).group(1)))
    age_matrix = df[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Baseline and last observed age
    age_last = np.nanmax(age_matrix, axis=1)
    age_first = np.array([
        age_matrix[i, np.where(np.isfinite(age_matrix[i]))[0][0]]
        if np.any(np.isfinite(age_matrix[i])) else np.nan
        for i in range(df.shape[0])
    ])

    features = {}

    # Static clinical features
    clinical_cols = ["gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]
    for col in clinical_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()

    features["age_last"] = age_last
    features["observation_duration"] = age_last - age_first
    features["visit_count"] = np.isfinite(age_matrix).sum(axis=1).astype(float)

    # Longitudinal biomarker processing
    for b_low, b_orig in biomarker_bases.items():
        vcols = [f"{b_orig}_v{i}" for i in range(1, 25) if f"{b_orig}_v{i}" in df.columns]
        if not vcols: continue

        biomarker_matrix = df[vcols].apply(pd.to_numeric, errors="coerce").to_numpy()
        visit_ages = age_matrix[:, [int(re.search(r"_v(\d+)$", c).group(1)) - 1 for c in vcols]]

        finite_mask = np.isfinite(biomarker_matrix)
        obs_count = finite_mask.sum(axis=1).astype(float)

        last_val = np.full(df.shape[0], np.nan)
        first_val = np.full(df.shape[0], np.nan)
        max_val = np.nanmax(biomarker_matrix, axis=1)
        slope = np.full(df.shape[0], np.nan)
        volatility = np.full(df.shape[0], np.nan)
        tsfa = np.full(df.shape[0], np.nan) # Time Since First Abnormal

        # Clinical thresholds for "abnormal" detection
        threshold = 40 if b_low in ["alt", "ast"] else 50 if b_low == "ggt" else 8.5 if b_low.startswith("fibs") else 0.5 if b_low.startswith("fibro") else None

        for i in range(df.shape[0]):
            idx = np.where(finite_mask[i])[0]
            if len(idx) > 0:
                last_val[i] = biomarker_matrix[i, idx[-1]]
                first_val[i] = biomarker_matrix[i, idx[0]]
                if len(idx) >= 2:
                    dt = visit_ages[i, idx[-1]] - visit_ages[i, idx[0]]
                    if dt > 0:
                        slope[i] = (biomarker_matrix[i, idx[-1]] - biomarker_matrix[i, idx[0]]) / dt
                    volatility[i] = np.std(np.diff(biomarker_matrix[i, idx]))

                if threshold:
                    is_abnormal = (biomarker_matrix[i] > threshold) if b_low != "plt" else (biomarker_matrix[i] < 150)
                    abnormal_idx = np.where(is_abnormal & finite_mask[i])[0]
                    if len(abnormal_idx) > 0:
                        tsfa[i] = age_last[i] - visit_ages[i, abnormal_idx[0]]

        features[f"{b_low}_last"] = last_val
        features[f"{b_low}_max"] = max_val
        features[f"{b_low}_slope"] = slope
        features[f"{b_low}_volatility"] = volatility
        features[f"{b_low}_tsfa"] = tsfa
        features[f"{b_low}_missing_rate"] = 1.0 - (obs_count / float(len(vcols)))

        if threshold:
            features[f"{b_low}_is_abnormal"] = (max_val > threshold).astype(float) if b_low != "plt" else (np.nanmin(biomarker_matrix, axis=1) < 150).astype(float)

    # Composite Clinical Scores (FIB-4, APRI)
    if all(k in features for k in ["ast_last", "alt_last", "plt_last"]):
        ast = features["ast_last"]
        alt = features["alt_last"]
        plt = features["plt_last"]
        features["fib4"] = (age_last * ast) / (plt * np.sqrt(np.maximum(alt, 1e-6)))
        features["apri"] = (ast / 40.0) * 100.0 / np.maximum(plt, 1e-6)
        features["ast_alt_ratio"] = ast / np.maximum(alt, 1e-6)

    result_df = pd.DataFrame(features, index=df.index).replace([np.inf, -np.inf], np.nan)
    # Drop columns with 100% missing values
    return result_df.dropna(axis=1, how='all')

# --- Training and Prediction Pipeline ---

def prepare_targets(df, outcome_type):
    """Parses raw data into scikit-survival compatible format."""
    event_col = "evenements_hepatiques_majeurs" if outcome_type == "hepatic" else "death"
    age_col = "evenements_hepatiques_age_occur" if outcome_type == "hepatic" else "death_age_occur"

    # Filter rows with missing labels
    mask = (~df[event_col].isna()) & (~((df[event_col] == 1) & df[age_col].isna()))
    valid_df = df.loc[mask].copy()

    age_cols = [c for c in df.columns if c.lower().startswith("age_v")]
    age_matrix = valid_df[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    baseline_age = np.array([age_matrix[i, np.where(np.isfinite(age_matrix[i]))[0][0]] for i in range(valid_df.shape[0])])

    event_occurred = (valid_df[event_col] == 1).to_numpy()
    survival_time = np.where(
        event_occurred,
        pd.to_numeric(valid_df[age_col], errors="coerce").to_numpy() - baseline_age,
        np.nanmax(age_matrix, axis=1) - baseline_age
    )

    return valid_df, Surv.from_arrays(event=event_occurred, time=np.maximum(survival_time, 0.001))

def get_ensemble_models(seed, n_estimators):
    """Initializes the collection of models for the ensemble."""
    return [
        SkSurvWrapper("RSF", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=12, random_state=seed, n_jobs=-1))
        ])),
        SkSurvWrapper("GBSA", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, learning_rate=0.03, max_depth=3, random_state=seed))
        ])),
        XGBoostSurvivalModel("XGB-Cox", {
            "objective": "survival:cox", "tree_method": "hist", "learning_rate": 0.008, "max_depth": 4, "seed": seed, "verbosity": 0
        }, n_estimators * 2),
        LightGBMSurvivalProxy("LGB-Proxy", {
            "objective": "regression", "learning_rate": 0.008, "num_leaves": 31, "verbose": -1, "seed": seed
        }, n_estimators * 2),
        CatBoostSurvivalModel("CAT-Cox", {
            "loss_function": "Cox", "learning_rate": 0.015, "depth": 4, "random_seed": seed
        }, n_estimators)
    ]

def rank_transform(predictions):
    """Converts absolute risk scores to percentile ranks for ensemble stability."""
    return pd.Series(predictions).rank(pct=True).to_numpy()

def run_annitia_pipeline(train_csv, test_csv, output_csv):
    """Main execution loop for model training and prediction."""
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract initial test features
    test_features_full = build_features(test_df)
    final_results = {}

    for outcome in ["hepatic", "death"]:
        print(f"\nTraining ensemble for {outcome} outcome...")
        target_train_df, y_train = prepare_targets(train_df, outcome)
        target_train_df = target_train_df.reset_index(drop=True)

        # Build features with truncation to prevent leakage
        event_age_col = "evenements_hepatiques_age_occur" if outcome == "hepatic" else "death_age_occur"
        train_features = build_features(target_train_df, target_train_df[event_age_col])

        # Match columns between train and test
        common_cols = train_features.columns.intersection(test_features_full.columns)
        X_train = train_features[common_cols]
        X_test = test_features_full[common_cols]

        # Cross-validation for weight optimization
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_models = 5
        oof_predictions = np.zeros((X_train.shape[0], n_models))

        for train_idx, val_idx in kf.split(X_train, y_train["event"]):
            Xt, yt = X_train.iloc[train_idx], y_train[train_idx]
            for i, model in enumerate(get_ensemble_models(42, 250)):
                model.fit(Xt, yt)
                oof_predictions[val_idx, i] = model.predict_risk(X_train.iloc[val_idx])

        # Optimize ensemble weights using Dirichlet search on OOF C-index
        oof_ranks = np.column_stack([rank_transform(oof_predictions[:, i]) for i in range(n_models)])
        best_weights, best_c_index = np.ones(n_models)/n_models, -1

        for _ in range(1000):
            weights = np.random.dirichlet(np.ones(n_models))
            score = concordance_index_censored(y_train["event"], y_train["time"], oof_ranks @ weights)[0]
            if score > best_c_index:
                best_c_index, best_weights = score, weights

        print(f"  Optimized OOF C-index: {best_c_index:.4f}")
        print(f"  Ensemble Weights: {best_weights.round(3)}")

        # Final Bagging across multiple seeds
        seed_predictions = []
        for bag_seed in [42, 1337, 2026]:
            print(f"  Bagging progress: seed {bag_seed}...")
            models = [m.fit(X_train, y_train) for m in get_ensemble_models(bag_seed, 500)]
            test_rank_matrix = np.column_stack([rank_transform(m.predict_risk(X_test)) for m in models])
            seed_predictions.append(test_rank_matrix @ best_weights)

        final_results[outcome] = np.mean(seed_predictions, axis=0)

    # Save the final submission
    submission = pd.DataFrame({
        "trustii_id": test_df["trustii_id"],
        "risk_hepatic_event": final_results["hepatic"],
        "risk_death": final_results["death"]
    })
    submission.sort_values("trustii_id").to_csv(output_csv, index=False)
    print(f"\nPipeline complete! Submission saved to {output_csv}")

if __name__ == "__main__":
    run_annitia_pipeline("competition_data/Train.csv", "competition_data/Test.csv", "submission.csv")
