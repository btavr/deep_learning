
"""
grader_breast_cancer.py

Full 100-point, detailed, colorized in-notebook self-check grader.

Usage in notebook:

    from grader_breast_cancer import grade
    grade(globals())

- Safe to run even if the notebook is partially completed.
- Produces a detailed breakdown + total /100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)

# -----------------------------------------------------
# Types + utilities
# -----------------------------------------------------

@dataclass
class Check:
    section: str
    name: str
    points: int
    ok: bool
    msg: str

def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _close(a: Optional[float], b: float, tol: float) -> bool:
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol

def _fmt(x: Any, digits: int = 3) -> str:
    xf = _safe_float(x)
    if xf is None:
        return "None"
    return f"{xf:.{digits}f}"

def _safe_shape(x: Any) -> str:
    return str(getattr(x, "shape", None))

# -----------------------------------------------------
# Reference solution (baseline)
# -----------------------------------------------------

def _baseline_reference() -> Dict[str, Any]:
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    test_accuracy = pipe.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    probs_pos = pipe.predict_proba(X_test)[:, 1]
    tau = 0.30
    y_pred_tau = (probs_pos >= tau).astype(int)
    recall_tau = recall_score(y_test, y_pred_tau)
    cm_tau = confusion_matrix(y_test, y_pred_tau)

    fpr, tpr, _ = roc_curve(y_test, probs_pos)
    roc_auc = auc(fpr, tpr)

    cv_ss = ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv_ss, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    best_params = grid.best_params_
    best_f1 = float(grid.best_score_)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "pipe": pipe,
        "test_accuracy": float(test_accuracy),
        "y_pred": y_pred,
        "cm": cm,
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "probs_pos": probs_pos,
        "tau": tau,
        "y_pred_tau": y_pred_tau,
        "recall_tau": float(recall_tau),
        "cm_tau": cm_tau,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(roc_auc),
        "cv_ss": cv_ss,
        "cv_scores": cv_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "param_grid": param_grid,
        "grid_scoring": "f1",
        "best_params": best_params,
        "best_f1": best_f1,
    }

_REF = _baseline_reference()

# -----------------------------------------------------
# Rendering (HTML in Jupyter, plain text fallback)
# -----------------------------------------------------

def _render_html(checks: List[Check], score: int, total: int) -> None:
    try:
        from IPython.display import display, HTML  # type: ignore
    except Exception:
        _render_text(checks, score, total)
        return

    pct = 100.0 * score / (total or 1)

    # Group checks by section
    sections: Dict[str, List[Check]] = {}
    for c in checks:
        sections.setdefault(c.section, []).append(c)

    def badge(ok: bool) -> str:
        if ok:
            return '<span style="color:#0a7a0a;font-weight:700;">PASS</span>'
        return '<span style="color:#b00020;font-weight:700;">FAIL</span>'

    rows = []
    for sec, items in sections.items():
        sec_score = sum(i.points for i in items if i.ok)
        sec_total = sum(i.points for i in items)
        rows.append(f"""
        <tr><td colspan="4" style="padding:10px 8px;background:#f6f6f6;font-weight:700;">
            {sec} — {sec_score}/{sec_total}
        </td></tr>
        """)
        for i in items:
            rows.append(f"""
            <tr>
              <td style="padding:6px 8px;white-space:nowrap;">{badge(i.ok)}</td>
              <td style="padding:6px 8px;white-space:nowrap;text-align:right;">{i.points}</td>
              <td style="padding:6px 8px;font-weight:600;">{i.name}</td>
              <td style="padding:6px 8px;color:#333;">{i.msg}</td>
            </tr>
            """)

    html = f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;">
      <div style="padding:10px 12px;border:1px solid #ddd;border-radius:10px;">
        <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;">
          <div style="font-size:18px;font-weight:800;">SKLEARN WORKSHEET — SELF CHECK</div>
          <div style="font-size:16px;">
            <span style="font-weight:800;">Total:</span>
            <span style="font-weight:800;">{score}/{total}</span>
            <span style="color:#666;">({pct:.1f}%)</span>
          </div>
        </div>
        <div style="margin-top:10px;">
          <table style="width:100%;border-collapse:collapse;border-top:1px solid #eee;">
            <thead>
              <tr>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #eee;">Status</th>
                <th style="text-align:right;padding:8px;border-bottom:1px solid #eee;">Pts</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #eee;">Item</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #eee;">Feedback</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
        <div style="margin-top:10px;color:#666;font-size:12px;">
          Note: This is a self-check for completion. If a valid approach is marked incorrect, check variable names and required random_state/stratify.
        </div>
      </div>
    </div>
    """
    display(HTML(html))

def _render_text(checks: List[Check], score: int, total: int) -> None:
    pct = 100.0 * score / (total or 1)
    print("\n" + "="*72)
    print("SKLEARN WORKSHEET — SELF CHECK")
    print("="*72)
    print(f"TOTAL: {score}/{total} ({pct:.1f}%)\n")

    # Group by section
    sections: Dict[str, List[Check]] = {}
    for c in checks:
        sections.setdefault(c.section, []).append(c)

    for sec, items in sections.items():
        sec_score = sum(i.points for i in items if i.ok)
        sec_total = sum(i.points for i in items)
        print(f"[{sec}] {sec_score}/{sec_total}")
        for i in items:
            status = "PASS" if i.ok else "FAIL"
            print(f"  {status:<4} | {i.points:>3} | {i.name}")
            print(f"       {i.msg}")
        print("-"*72)

    print("Note: If something is marked incorrect but your approach is valid,")
    print("      check variable names and required random_state/stratify.\n")

# -----------------------------------------------------
# Grade
# -----------------------------------------------------

def grade(ns: Dict[str, Any]) -> None:
    """
    Prints a detailed, colorized breakdown (HTML in Jupyter) and total score /100.
    """
    checks: List[Check] = []
    score = 0

    def add(section: str, name: str, points: int, ok: bool, msg_ok: str, msg_bad: str):
        nonlocal score
        checks.append(Check(section=section, name=name, points=points, ok=ok,
                            msg=(msg_ok if ok else msg_bad)))
        if ok:
            score += points

    # =================================================
    # Section 1: Load dataset (8)
    # =================================================
    X = ns.get("X", None)
    y = ns.get("y", None)

    add("1) Load dataset", "X loaded with correct shape", 4,
        _is_array(X) and getattr(X, "shape", None) == _REF["X"].shape,
        f"Shape: {_safe_shape(X)}",
        "Missing/incorrect X. Expected numpy array shape (569, 30).")

    add("1) Load dataset", "y loaded with correct shape", 4,
        _is_array(y) and getattr(y, "shape", None) == _REF["y"].shape,
        f"Shape: {_safe_shape(y)}",
        "Missing/incorrect y. Expected numpy array shape (569,).")

    # =================================================
    # Section 2: Train/test split (10)
    # =================================================
    X_train = ns.get("X_train", None)
    X_test = ns.get("X_test", None)
    y_train = ns.get("y_train", None)
    y_test = ns.get("y_test", None)

    ok_split = (
        _is_array(X_train) and _is_array(X_test) and _is_array(y_train) and _is_array(y_test)
        and getattr(X_train, "shape", None) == _REF["X_train"].shape
        and getattr(X_test, "shape", None) == _REF["X_test"].shape
        and getattr(y_train, "shape", None) == _REF["y_train"].shape
        and getattr(y_test, "shape", None) == _REF["y_test"].shape
    )
    add("2) Train/test split", "Stratified split shapes correct", 10,
        ok_split,
        f"Train {_safe_shape(X_train)}, Test {_safe_shape(X_test)}",
        "Split missing/incorrect. Use train_test_split(test_size=0.2, random_state=42, stratify=y).")

    # =================================================
    # Section 3: Pipeline + training (20)
    # =================================================
    pipe = ns.get("pipe", None)

    ok_pipe = (
        isinstance(pipe, Pipeline)
        and list(getattr(pipe, "named_steps", {}).keys()) == ["scaler", "clf"]
        and isinstance(pipe.named_steps.get("scaler"), StandardScaler)
        and isinstance(pipe.named_steps.get("clf"), LogisticRegression)
    )
    add("3) Pipeline + train", "Pipeline: StandardScaler + LogisticRegression", 10,
        ok_pipe,
        "Pipeline structure OK.",
        "pipe missing/incorrect. Expected Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(...))]).")

    test_accuracy = _safe_float(ns.get("test_accuracy", None))
    add("3) Pipeline + train", "test_accuracy computed (close to reference)", 6,
        test_accuracy is not None and _close(test_accuracy, _REF["test_accuracy"], tol=0.03),
        f"test_accuracy={_fmt(test_accuracy)}",
        "test_accuracy missing/incorrect. Use pipe.score(X_test, y_test).")

    y_pred = ns.get("y_pred", None)
    add("3) Pipeline + train", "y_pred computed with correct shape", 4,
        _is_array(y_pred) and getattr(y_pred, "shape", None) == _REF["y_pred"].shape,
        f"Shape: {_safe_shape(y_pred)}",
        "y_pred missing/incorrect. Use pipe.predict(X_test).")

    # =================================================
    # Section 4: Confusion matrix + metrics (15)
    # =================================================
    cm = ns.get("cm", None)
    add("4) Evaluation metrics", "Confusion matrix cm is 2x2", 5,
        _is_array(cm) and getattr(cm, "shape", None) == (2, 2),
        "cm is 2x2.",
        "cm missing/incorrect. Use confusion_matrix(y_test, y_pred).")

    test_precision = _safe_float(ns.get("test_precision", None))
    test_recall = _safe_float(ns.get("test_recall", None))
    test_f1 = _safe_float(ns.get("test_f1", None))

    ok_metrics = (
        test_precision is not None and test_recall is not None and test_f1 is not None
        and _close(test_precision, _REF["test_precision"], tol=0.05)
        and _close(test_recall, _REF["test_recall"], tol=0.05)
        and _close(test_f1, _REF["test_f1"], tol=0.05)
    )
    add("4) Evaluation metrics", "precision/recall/f1 computed (close to reference)", 10,
        ok_metrics,
        f"precision={_fmt(test_precision)}, recall={_fmt(test_recall)}, f1={_fmt(test_f1)}",
        "Missing/incorrect precision/recall/f1. Use precision_score/recall_score/f1_score(y_test, y_pred).")

    # =================================================
    # Section 5: Threshold adjustment (15)
    # =================================================
    probs_pos = ns.get("probs_pos", None)
    ok_probs = (
        _is_array(probs_pos)
        and getattr(probs_pos, "shape", None) == _REF["probs_pos"].shape
        and np.all((probs_pos >= 0) & (probs_pos <= 1))
    )
    add("5) Thresholding", "probs_pos computed (predict_proba)", 6,
        ok_probs,
        f"Shape: {_safe_shape(probs_pos)} (values in [0,1])",
        "probs_pos missing/incorrect. Use pipe.predict_proba(X_test)[:,1].")

    tau = ns.get("tau", None)
    add("5) Thresholding", "tau = 0.30", 2,
        tau is not None and _close(_safe_float(tau), 0.30, tol=1e-9),
        "tau=0.30",
        "tau missing/incorrect. Set tau = 0.30.")

    y_pred_tau = ns.get("y_pred_tau", None)
    add("5) Thresholding", "y_pred_tau computed with correct shape", 3,
        _is_array(y_pred_tau) and getattr(y_pred_tau, "shape", None) == _REF["y_pred_tau"].shape,
        f"Shape: {_safe_shape(y_pred_tau)}",
        "y_pred_tau missing/incorrect. Use (probs_pos >= tau).astype(int).")

    recall_tau = _safe_float(ns.get("recall_tau", None))
    cm_tau = ns.get("cm_tau", None)
    add("5) Thresholding", "recall_tau + cm_tau computed", 4,
        recall_tau is not None and _is_array(cm_tau) and getattr(cm_tau, "shape", None) == (2, 2) and 0 <= float(recall_tau) <= 1,
        f"recall_tau={_fmt(recall_tau)}, cm_tau=2x2",
        "recall_tau/cm_tau missing/incorrect. Use recall_score(y_test, y_pred_tau) and confusion_matrix(y_test, y_pred_tau).")

    # =================================================
    # Section 6: ROC + AUC (10)
    # =================================================
    fpr = ns.get("fpr", None)
    tpr = ns.get("tpr", None)
    roc_auc = _safe_float(ns.get("roc_auc", None))
    add("6) ROC/AUC", "ROC AUC computed (close to reference)", 10,
        _is_array(fpr) and _is_array(tpr) and roc_auc is not None and 0.0 <= float(roc_auc) <= 1.0 and _close(roc_auc, _REF["roc_auc"], tol=0.05),
        f"roc_auc={_fmt(roc_auc)}",
        "ROC/AUC missing/incorrect. Use fpr,tpr,_=roc_curve(y_test, probs_pos) and roc_auc=auc(fpr,tpr).")

    # =================================================
    # Section 7: Cross-validation (12)
    # =================================================
    cv_ss = ns.get("cv_ss", None)
    ok_ss = isinstance(cv_ss, ShuffleSplit) and getattr(cv_ss, "n_splits", None) == 20
    add("7) Cross-validation", "ShuffleSplit defined (n_splits=20)", 4,
        ok_ss,
        "cv_ss OK.",
        "cv_ss missing/incorrect. Use ShuffleSplit(n_splits=20, test_size=0.2, random_state=42).")

    cv_scores = ns.get("cv_scores", None)
    add("7) Cross-validation", "cv_scores computed (correct shape)", 4,
        _is_array(cv_scores) and getattr(cv_scores, "shape", None) == _REF["cv_scores"].shape,
        f"Shape: {_safe_shape(cv_scores)}",
        "cv_scores missing/incorrect. Use cross_val_score(pipe, X, y, cv=cv_ss, scoring='accuracy').")

    cv_mean = _safe_float(ns.get("cv_mean", None))
    cv_std = _safe_float(ns.get("cv_std", None))
    add("7) Cross-validation", "cv_mean and cv_std computed", 4,
        (cv_mean is not None and cv_std is not None and 0 <= float(cv_mean) <= 1 and float(cv_std) >= 0),
        f"cv_mean={_fmt(cv_mean)}, cv_std={_fmt(cv_std)}",
        "cv_mean/cv_std missing/incorrect. Use mean/std of cv_scores.")

    # =================================================
    # Section 8: GridSearchCV (10)
    # =================================================
    param_grid = ns.get("param_grid", None)
    add("8) Hyperparameter search", "param_grid has clf__C values", 3,
        isinstance(param_grid, dict) and "clf__C" in param_grid and list(param_grid["clf__C"]) == [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_grid OK.",
        "param_grid missing/incorrect. Expected {'clf__C':[0.01,0.1,1.0,10.0,100.0]}.")

    grid = ns.get("grid", None)
    add("8) Hyperparameter search", "GridSearchCV object created", 3,
        isinstance(grid, GridSearchCV) and getattr(grid, "scoring", None) == "f1",
        "grid OK.",
        "grid missing/incorrect. Create GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1).")

    best_params = ns.get("best_params", None)
    best_f1 = _safe_float(ns.get("best_f1", None))
    add("8) Hyperparameter search", "best_params and best_f1 reported", 4,
        isinstance(best_params, dict) and "clf__C" in best_params and best_f1 is not None and 0 <= float(best_f1) <= 1,
        f"best_params={best_params}, best_f1={_fmt(best_f1)}",
        "best_params/best_f1 missing/incorrect. After fit: best_params=grid.best_params_, best_f1=grid.best_score_.")

    # -------------------------------------------------
    # Final score sanity: should be /100
    # -------------------------------------------------
    total = sum(c.points for c in checks)
    # If someone edits point allocations, still display consistent totals
    if total != 100:
        # Normalize (rare). Keep stable behavior.
        # This only changes the final displayed percentage, not per-item points.
        pass

    # Render
    _render_html(checks, score, total)
