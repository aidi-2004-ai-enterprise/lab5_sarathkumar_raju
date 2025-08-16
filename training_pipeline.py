# training_pipeline.py
"""
Lab 5 — Complete pipeline: EDA → Preprocessing → Feature Selection → Tuning →
Model Training → Evaluation → SHAP → PSI + Markdown report.

Run:
    python training_pipeline.py \
        --data_csv path/to/bankruptcy.csv \
        --target "Bankrupt?" \
        --out_dir artifacts

Notes / Key choices:
- Imbalance: tune with PR-AUC (average_precision) + class_weight="balanced" where applicable
  and XGBoost scale_pos_weight = (neg/pos).
- Preprocessing: median imputation (train stats), IQR winsorization (train caps),
  correlation filter (|r|>0.90), StandardScaler (train stats) for LR/XGB.
- Feature selection: simple correlation filter for stability & interpretability.
- Tuning: RandomizedSearchCV + StratifiedKFold(n_splits=5, shuffle=True).
- Interpretability: SHAP TreeExplainer for best XGB only (others skipped for speed).
- Drift: PSI (train vs test) bar chart + top-k feature hist overlays.
- Outputs: all charts + a Markdown report in --out_dir.

Author: Your Name
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# XGBoost is optional dependency; we use it if available.
try:
    from xgboost import XGBClassifier

    _HAVE_XGB = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_XGB = False

# SHAP is used only for the selected XGB model; we guard imports at use time.
RANDOM_STATE: int = 42
N_JOBS: int = -1


# ------------------------------ Utilities ---------------------------------- #
def configure_logging(out_dir: Path) -> None:
    """Set up logging to file and console."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def savefig(fig_path: Path) -> None:
    """Save the current matplotlib figure to `fig_path` safely."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()


def iqr_caps(series: pd.Series) -> Tuple[float, float]:
    """Return (lower_cap, upper_cap) based on IQR winsorization."""
    q1, q3 = np.nanpercentile(series, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def winsorize_df(
    df: pd.DataFrame, caps: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """Apply winsorization caps per column. Assumes numeric df."""
    clipped = df.copy()
    for c, (lo, hi) in caps.items():
        clipped[c] = clipped[c].clip(lower=lo, upper=hi)
    return clipped


def correlation_filter(X: pd.DataFrame, threshold: float = 0.90) -> List[str]:
    """Drop highly correlated columns (> threshold). Return remaining feature names."""
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in X.columns if c not in to_drop]


def psi(expected: ArrayLike, actual: ArrayLike, bins: int = 10) -> float:
    """
    Compute Population Stability Index for one feature.
    expected: reference (e.g., train), actual: comparison (e.g., test).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    # Protect against identical quantiles -> add tiny noise
    edges = np.unique(edges)
    if edges.size < bins + 1:
        edges = np.linspace(np.min(expected), np.max(expected), bins + 1)
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, 1)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def top_k_psi(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 15) -> pd.Series:
    """Compute PSI for all numeric columns and return top-k descending."""
    psis: Dict[str, float] = {}
    for col in train_df.columns:
        try:
            psis[col] = psi(train_df[col].values, test_df[col].values)
        except Exception:
            continue
    s = pd.Series(psis).sort_values(ascending=False)
    return s.head(k)


# ---------------------------- Data & EDA ----------------------------------- #
def quick_eda(df: pd.DataFrame, target: str, out_dir: Path) -> Dict[str, Any]:
    """Generate a few simple EDA artifacts and return short stats."""
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Class balance
    y = df[target].values
    pos_rate = float(np.mean(y))
    plt.figure()
    counts = pd.Series(y).value_counts().sort_index()
    counts.index = ["Class 0", "Class 1"]
    counts.plot(kind="bar")
    plt.title("Class Balance")
    plt.ylabel("Count")
    savefig(eda_dir / "class_balance.png")

    # Correlation heatmap (subset if too wide)
    numeric = df.drop(columns=[target]).select_dtypes(include=[np.number])
    subset = numeric.sample(n=min(25, numeric.shape[1]), random_state=RANDOM_STATE)
    corr = subset.corr().abs()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.title("Correlation Heatmap (subset)")
    plt.colorbar()
    savefig(eda_dir / "corr_subset_heatmap.png")

    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "pos_rate": pos_rate}


# ---------------------------- Modeling ------------------------------------- #
@dataclass
class ModelResult:
    name: str
    estimator: Any
    best_params: Dict[str, Any]
    metrics_train: Dict[str, float]
    metrics_test: Dict[str, float]
    clf_report_test: str
    conf_mat_test: List[List[int]]
    proba_train: np.ndarray
    proba_test: np.ndarray


def build_search_spaces(
    scale_pos_weight: float,
) -> Dict[str, Tuple[Any, Dict[str, Iterable[Any]]]]:
    """Return dict of model name -> (estimator, param_distributions)."""
    models: Dict[str, Tuple[Any, Dict[str, Iterable[Any]]]] = {}

    # Logistic Regression
    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    models["LogReg"] = (
        lr,
        {
            "logreg__C": np.logspace(-3, 2, 20),
            "logreg__penalty": ["l1", "l2"],
        },
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    models["RandomForest"] = (
        rf,
        {
            "rf__max_depth": [None, 5, 10, 15, 20],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2", None],
        },
    )

    # XGBoost (optional)
    if _HAVE_XGB:
        xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )
        models["XGBoost"] = (
            xgb,
            {
                "xgb__max_depth": [3, 4, 5, 6, 8],
                "xgb__min_child_weight": [1, 3, 5, 7],
                "xgb__gamma": [0, 0.5, 1.0],
                "xgb__reg_alpha": [0, 0.001, 0.01, 0.1],
                "xgb__reg_lambda": [0.1, 1.0, 5.0, 10.0],
            },
        )

    return models


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute a common set of metrics from predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }
    return {k: float(v) for k, v in metrics.items()}


def plot_roc_pr_curves(
    results: List[ModelResult], y_test: np.ndarray, out_dir: Path
) -> None:
    """Create ROC and PR curves comparing all models."""
    # ROC
    plt.figure()
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res.proba_test)
        plt.plot(fpr, tpr, label=f"{res.name} (AUC={res.metrics_test['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves (test)")
    plt.legend()
    savefig(out_dir / "roc_curves.png")

    # PR
    plt.figure()
    for res in results:
        precision, recall, _ = precision_recall_curve(y_test, res.proba_test)
        plt.plot(
            recall, precision, label=f"{res.name} (AP={res.metrics_test['pr_auc']:.3f})"
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (test)")
    plt.legend()
    savefig(out_dir / "pr_curves.png")


def plot_calibration(
    results: List[ModelResult], y_test: np.ndarray, out_dir: Path
) -> None:
    """Plot reliability curves for each model (test)."""
    plt.figure()
    for res in results:
        prob_true, prob_pred = calibration_curve(y_test, res.proba_test, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=res.name)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves (test)")
    plt.legend()
    savefig(out_dir / "calibration_curves.png")


def plot_top_psi(psi_series: pd.Series, out_dir: Path) -> None:
    """Bar chart of top-k PSI features."""
    plt.figure(figsize=(8, 5))
    psi_series[::-1].plot(kind="barh")  # ascending for display
    plt.axvline(0.1, linestyle="--", label="0.1 (modest)")
    plt.axvline(0.25, linestyle="--", label="0.25 (significant)")
    plt.title("Top Feature Drift by PSI (train→test)")
    plt.xlabel("PSI")
    plt.legend()
    savefig(out_dir / "psi_topk.png")


def shap_summary_for_xgb(
    model: Any, X_train: pd.DataFrame, out_dir: Path
) -> Optional[Path]:
    """
    Generate SHAP summary for an XGBoost model if SHAP is available.
    Returns path to saved image or None.
    """
    try:
        import shap  # type: ignore
    except Exception:  # pragma: no cover - optional
        logging.info("SHAP not available; skipping.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        path = out_dir / "shap_summary.png"
        savefig(path)
        return path
    except Exception as e:  # pragma: no cover - robustness
        logging.info("SHAP failed: %s", e)
        return None


# ------------------------------ Pipeline ----------------------------------- #
def run_pipeline(data_csv: Path, target: str, out_dir: Path) -> None:
    """Main orchestration function."""
    configure_logging(out_dir)
    logging.info("Loading data: %s", data_csv)
    df = pd.read_csv(data_csv)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV.")

    # Ensure target is 0/1 integers
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    # EDA
    stats = quick_eda(df, target, out_dir)
    logging.info("Rows × Cols: %s × %s", stats["rows"], stats["cols"])
    logging.info("Target positive rate: %.4f", stats["pos_rate"])

    # Split
    X = df.drop(columns=[target])
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessing (numeric only dataset assumed)
    num_cols = X_train.columns.tolist()

    # Median imputer fit on train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=num_cols, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=num_cols, index=X_test.index
    )

    # IQR winsorization caps from train, then apply to both
    caps = {c: iqr_caps(X_train_imp[c]) for c in num_cols}
    X_train_win = winsorize_df(X_train_imp, caps)
    X_test_win = winsorize_df(X_test_imp, caps)

    # Correlation filter based on train
    selected = correlation_filter(X_train_win, threshold=0.90)
    pd.Series(selected).to_csv(
        out_dir / "selected_features.csv", index=False, header=False
    )
    X_train_sel = X_train_win[selected]
    X_test_sel = X_test_win[selected]

    # Scaler (for LR & XGB); RF is tree-based but we’ll share the same input
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_sel), columns=selected, index=X_train_sel.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_sel), columns=selected, index=X_test_sel.index
    )

    # Class imbalance helper
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = float(neg / max(pos, 1))

    # Build models & param spaces
    spaces = build_search_spaces(scale_pos_weight=spw)

    # Prepare pipelines for each model
    results: List[ModelResult] = []
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (estimator, params) in spaces.items():
        if name == "LogReg":
            pipe = Pipeline([("logreg", estimator)])
            search_params = params
            Xtr, Xte = X_train_scaled, X_test_scaled
        elif name == "RandomForest":
            pipe = Pipeline([("rf", estimator)])
            search_params = params
            # RF is robust to scale; use unscaled to keep raw distributions
            Xtr, Xte = X_train_sel, X_test_sel
        else:  # XGBoost
            pipe = Pipeline([("xgb", estimator)])
            search_params = params
            Xtr, Xte = X_train_scaled, X_test_scaled

        logging.info("Tuning %s ...", name)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=search_params,
            n_iter=25,
            scoring="average_precision",
            n_jobs=N_JOBS,
            cv=cv,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        search.fit(Xtr, y_train)
        best_est = search.best_estimator_

        # Save best model
        model_path = out_dir / f"{name}_best.joblib"
        joblib.dump(best_est, model_path)

        # Probabilities
        proba_tr = best_est.predict_proba(Xtr)[:, 1]
        proba_te = best_est.predict_proba(Xte)[:, 1]

        # Metrics
        m_tr = evaluate_probs(y_train, proba_tr)
        m_te = evaluate_probs(y_test, proba_te)
        report_te = classification_report(y_test, (proba_te >= 0.5).astype(int))
        cm_te = confusion_matrix(y_test, (proba_te >= 0.5).astype(int)).tolist()

        results.append(
            ModelResult(
                name=name,
                estimator=best_est,
                best_params=search.best_params_,
                metrics_train=m_tr,
                metrics_test=m_te,
                clf_report_test=report_te,
                conf_mat_test=cm_te,
                proba_train=proba_tr,
                proba_test=proba_te,
            )
        )

        logging.info(
            "%s best params: %s", name, json.dumps(search.best_params_, indent=2)
        )
        logging.info(
            "%s test PR-AUC: %.4f | ROC-AUC: %.4f",
            name,
            m_te["pr_auc"],
            m_te["roc_auc"],
        )

    # Plots
    plot_roc_pr_curves(results, y_test, curves_dir)
    plot_calibration(results, y_test, curves_dir)

    # PSI (train vs test) for selected features (unscaled, winsorized)
    psi_series = top_k_psi(X_train_sel, X_test_sel, k=15)
    drift_dir = out_dir / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    psi_series.to_csv(drift_dir / "psi_topk.csv", header=["psi"])
    plot_top_psi(psi_series, drift_dir)

    # SHAP for best XGB (if present and installed)
    shap_path: Optional[Path] = None
    xgb_res = next((r for r in results if r.name == "XGBoost"), None)
    if xgb_res is not None:
        try:
            # Extract inner model if using a pipeline
            xgb_core = xgb_res.estimator.named_steps["xgb"]
            shap_path = shap_summary_for_xgb(xgb_core, X_train_scaled, out_dir / "shap")
        except Exception as e:
            logging.info("Skipping SHAP due to: %s", e)

    # Build comparison table
    comp_rows: List[Dict[str, Any]] = []
    for r in results:
        row = {
            "model": r.name,
            "train_pr_auc": r.metrics_train["pr_auc"],
            "test_pr_auc": r.metrics_test["pr_auc"],
            "train_roc_auc": r.metrics_train["roc_auc"],
            "test_roc_auc": r.metrics_test["roc_auc"],
            "test_brier": r.metrics_test["brier"],
            "test_accuracy": r.metrics_test["accuracy"],
        }
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows).sort_values("test_pr_auc", ascending=False)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    # Markdown report
    report_md = out_dir / "REPORT.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Lab 5 Report — Training Pipeline\n\n")
        f.write(f"- Rows × Columns: **{stats['rows']} × {stats['cols']}**\n")
        f.write(f"- Target positive rate: **{stats['pos_rate']:.4f}**\n")
        f.write(f"- Selected features: **{len(selected)}/{X.shape[1]}**\n\n")

        f.write("## EDA\n")
        f.write(
            "- Target imbalance visualized; correlations inspected for redundancy.\n\n"
        )
        f.write("![Class Balance](eda/class_balance.png)\n\n")
        f.write("![Correlation (subset)](eda/corr_subset_heatmap.png)\n\n")

        f.write("## Data Preprocessing\n")
        f.write(
            "- Train-median imputation → IQR winsorization → Correlation filter (|r|>0.90) "
            "→ StandardScaler (for LR/XGB).\n"
        )
        f.write("- Selected features saved to `selected_features.csv`.\n\n")

        f.write("## Hyperparameter Tuning\n")
        f.write(
            "- RandomizedSearchCV (n_iter=25), scoring=PR-AUC, CV=StratifiedKFold(5).\n\n"
        )

        f.write("## Model Comparison (sorted by Test PR-AUC)\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Curves & Calibration\n\n")
        f.write("![ROC](curves/roc_curves.png)\n\n")
        f.write("![PR](curves/pr_curves.png)\n\n")
        f.write("![Calibration](curves/calibration_curves.png)\n\n")

        f.write("## Confusion Matrices & Reports (Test)\n\n")
        for r in results:
            f.write(f"### {r.name}\n\n")
            f.write("**Confusion Matrix (test)**\n\n")
            f.write("```\n")
            f.write(json.dumps(r.conf_mat_test))
            f.write("\n```\n\n")
            f.write("**Classification Report (test)**\n\n```\n")
            f.write(r.clf_report_test)
            f.write("\n```\n\n")

        f.write("## Drift — Population Stability Index (PSI)\n\n")
        f.write("Top features by PSI (train→test): `drift/psi_topk.csv`\n\n")
        f.write("![PSI](drift/psi_topk.png)\n\n")

        if shap_path is not None:
            f.write("## SHAP (XGBoost)\n\n")
            f.write("![SHAP Summary](shap/shap_summary.png)\n\n")
        else:
            f.write(
                "## SHAP (XGBoost)\n\n- Skipped (XGBoost or SHAP not available).\n\n"
            )

        f.write("## Reproducibility\n")
        f.write("- Random seeds fixed; dependencies documented in project.\n")

    # Also dump a compact JSON of key results
    summary = {
        "rows": stats["rows"],
        "cols": stats["cols"],
        "pos_rate": stats["pos_rate"],
        "selected_features": len(selected),
        "models": [
            {
                "name": r.name,
                "best_params": r.best_params,
                "test_pr_auc": r.metrics_test["pr_auc"],
                "test_roc_auc": r.metrics_test["roc_auc"],
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Done. Artifacts written to: %s", str(out_dir))


# ------------------------------- CLI --------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Lab 5: Model Implementation Pipeline")
    p.add_argument("--data_csv", type=Path, required=True, help="Input CSV path")
    p.add_argument("--target", type=str, default="Bankrupt?", help="Target column name")
    p.add_argument(
        "--out_dir", type=Path, default=Path("artifacts"), help="Artifacts dir"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data_csv, args.target, args.out_dir)
