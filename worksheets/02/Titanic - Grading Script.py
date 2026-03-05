"""
grader_titanic.py

Full 100-point, detailed, colorized in-notebook self-check grader
for the Titanic worksheet (Xa numeric embarked encoding vs Xb one-hot).

Usage in notebook:

    from grader_titanic import grade
    grade(globals())

- Safe to run even if notebook is partially completed.
- Produces a detailed breakdown + total /100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


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

def _is_df(x: Any) -> bool:
    return isinstance(x, pd.DataFrame)

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

def _file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


# -----------------------------------------------------
# Reference solution (baseline)
# -----------------------------------------------------

def _load_titanic() -> pd.DataFrame:
    # Try TSV+decimal comma, then plain CSV
    try:
        df = pd.read_csv("titanic.csv", sep="\t", decimal=",")
        if "survived" not in df.columns:
            raise ValueError("Missing 'survived' after TSV read")
        return df
    except Exception:
        df = pd.read_csv("titanic.csv")
        return df

def _impute_reference(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    # embarked: missing -> "C"
    if "embarked" in tmp.columns:
        tmp.loc[tmp["embarked"].isna(), "embarked"] = "C"

    # fare: missing -> median fare among pclass==3
    if "fare" in tmp.columns and "pclass" in tmp.columns:
        med3 = tmp.loc[tmp["pclass"] == 3, "fare"].median()
        tmp.loc[tmp["fare"].isna(), "fare"] = med3

    # age: group-wise median by sex & pclass
    if "age" in tmp.columns and "sex" in tmp.columns and "pclass" in tmp.columns:
        grp_med = tmp.groupby(["sex", "pclass"])["age"].transform("median")
        tmp["age"] = tmp["age"].fillna(grp_med)
        # final fallback if anything remains
        tmp["age"] = tmp["age"].fillna(tmp["age"].median())

    return tmp

def _build_Xa_Xb_y(df_imp: pd.DataFrame) -> Dict[str, Any]:
    # sex encoding
    sex_num = df_imp["sex"].map({"male": 0, "female": 1}).to_numpy()

    age = df_imp["age"].to_numpy()
    sibsp = df_imp["sibsp"].to_numpy()
    parch = df_imp["parch"].to_numpy()
    fare = df_imp["fare"].to_numpy()

    # Xa: embarked_num integer encoding
    embarked_num = df_imp["embarked"].map({"C": 0, "Q": 1, "S": 2}).to_numpy()

    Xa = np.column_stack([sex_num, age, sibsp, parch, fare, embarked_num])

    # Xb: one-hot with one column dropped (C baseline)
    emb_Q = (df_imp["embarked"] == "Q").astype(int).to_numpy()
    emb_S = (df_imp["embarked"] == "S").astype(int).to_numpy()
    Xb = np.column_stack([sex_num, age, sibsp, parch, fare, emb_Q, emb_S])

    y = df_imp["survived"].to_numpy()

    return {"Xa": Xa, "Xb": Xb, "y": y}

def _fit_reference_models(Xa: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    random_state = 42
    Xa_train, Xa_test, y_train, y_test = train_test_split(
        Xa, y, test_size=0.2, random_state=random_state, stratify=y
    )
    Xb_train, Xb_test, _, _ = train_test_split(
        Xb, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model_Xa = LogisticRegression(max_iter=2000)
    model_Xb = LogisticRegression(max_iter=2000)

    model_Xa.fit(Xa_train, y_train)
    model_Xb.fit(Xb_train, y_train)

    pXa = model_Xa.predict_proba(Xa_test)[:, 1]
    pXb = model_Xb.predict_proba(Xb_test)[:, 1]

    auc_Xa = float(roc_auc_score(y_test, pXa))
    auc_Xb = float(roc_auc_score(y_test, pXb))

    yhat_Xb = (pXb >= 0.5).astype(int)
    cm_Xb = confusion_matrix(y_test, yhat_Xb)

    return {
        "random_state": random_state,
        "Xa_train": Xa_train, "Xa_test": Xa_test,
        "Xb_train": Xb_train, "Xb_test": Xb_test,
        "y_train": y_train, "y_test": y_test,
        "model_Xa": model_Xa, "model_Xb": model_Xb,
        "auc_Xa": auc_Xa, "auc_Xb": auc_Xb,
        "cm_Xb": cm_Xb,
    }

def _baseline_reference() -> Dict[str, Any]:
    df = _load_titanic()
    df_imp = _impute_reference(df)
    mats = _build_Xa_Xb_y(df_imp)
    fitted = _fit_reference_models(mats["Xa"], mats["Xb"], mats["y"])
    return {
        "df": df,
        "df_imp": df_imp,
        **mats,
        **fitted,
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
          <div style="font-size:18px;font-weight:800;">TITANIC WORKSHEET — SELF CHECK</div>
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
    print("\n" + "=" * 72)
    print("TITANIC WORKSHEET — SELF CHECK")
    print("=" * 72)
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
        print("-" * 72)

    print("Note: If something is marked incorrect but your approach is valid,")
    print("      check variable names and required random_state/stratify.\n")


# -----------------------------------------------------
# Grade (100 points)
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
    # Section 1: Load + basic objects (8)
    # =================================================
    df = ns.get("df", None)
    add("1) Load data", "df loaded as DataFrame", 4,
        _is_df(df),
        f"df type OK. shape={getattr(df, 'shape', None)}",
        "df missing/incorrect. Expected pandas DataFrame named df.")

    add("1) Load data", "df contains 'survived'", 4,
        _is_df(df) and ("survived" in df.columns),
        "'survived' column present.",
        "Missing 'survived' column. Check file read options.")

    # =================================================
    # Section 2: Spain filter (8)
    # =================================================
    df_spain = ns.get("df_spain", None)
    if _is_df(df_spain):
        # Reference count
        mask_ref = _REF["df"]["home.dest"].astype(str).str.contains("spain", case=False, na=False)
        n_ref = int(mask_ref.sum())
        try:
            n_stu = int(df_spain.shape[0])
        except Exception:
            n_stu = -1

        ok_exact = (n_stu == n_ref)
        ok_nonempty = (n_ref == 0 and n_stu == 0) or (n_ref > 0 and n_stu > 0)

        add("2) Filtering", "df_spain exists (DataFrame)", 2,
            True, f"df_spain shape={df_spain.shape}", "df_spain missing.")

        add("2) Filtering", "Spain filter count matches reference", 6,
            ok_exact,
            f"OK: found {n_stu} (ref {n_ref})",
            f"Mismatch: found {n_stu} (ref {n_ref}). Use home.dest contains('spain', case=False).")
    else:
        add("2) Filtering", "df_spain exists (DataFrame)", 2, False,
            "df_spain OK.", "df_spain missing/incorrect.")
        add("2) Filtering", "Spain filter count matches reference", 6, False,
            "OK.", "Cannot check count without df_spain.")

    # =================================================
    # Section 3: Survival rates (12)
    # =================================================
    # Expect: rate_female, rate_male, rate_class1
    rate_female = _safe_float(ns.get("rate_female", None))
    rate_male = _safe_float(ns.get("rate_male", None))
    rate_class1 = _safe_float(ns.get("rate_class1", None))

    # reference values computed from raw df (not imputed needed)
    y_ref = _REF["df"]["survived"].to_numpy()
    sex_ref = _REF["df"]["sex"].astype(str).to_numpy()
    pclass_ref = _REF["df"]["pclass"].to_numpy()
    rf_ref = float(y_ref[sex_ref == "female"].mean())
    rm_ref = float(y_ref[sex_ref == "male"].mean())
    rc1_ref = float(y_ref[pclass_ref == 1].mean())

    add("3) EDA stats", "rate_female (close to reference)", 4,
        _close(rate_female, rf_ref, tol=0.03),
        f"rate_female={_fmt(rate_female)}",
        f"rate_female missing/incorrect. Expected ~{_fmt(rf_ref)}")

    add("3) EDA stats", "rate_male (close to reference)", 4,
        _close(rate_male, rm_ref, tol=0.03),
        f"rate_male={_fmt(rate_male)}",
        f"rate_male missing/incorrect. Expected ~{_fmt(rm_ref)}")

    add("3) EDA stats", "rate_class1 (close to reference)", 4,
        _close(rate_class1, rc1_ref, tol=0.03),
        f"rate_class1={_fmt(rate_class1)}",
        f"rate_class1 missing/incorrect. Expected ~{_fmt(rc1_ref)}")

    # =================================================
    # Section 4: Histogram file (8)
    # =================================================
    add("4) Plotting", "fig_age_hist.png saved", 8,
        _file_exists("fig_age_hist.png"),
        "Found fig_age_hist.png",
        "Missing fig_age_hist.png. Save the histogram figure with that filename.")

    # =================================================
    # Section 5: Imputation (20)
    # =================================================
    df_imp = ns.get("df_imp", None)
    add("5) Imputation", "df_imp exists (DataFrame)", 4,
        _is_df(df_imp),
        f"df_imp shape={getattr(df_imp,'shape',None)}",
        "df_imp missing/incorrect. Create df_imp = df.copy() then impute.")

    if _is_df(df_imp):
        emb_miss = int(df_imp["embarked"].isna().sum()) if "embarked" in df_imp.columns else -1
        fare_miss = int(df_imp["fare"].isna().sum()) if "fare" in df_imp.columns else -1
        age_miss = int(df_imp["age"].isna().sum()) if "age" in df_imp.columns else -1

        add("5) Imputation", "embarked has no missing", 5,
            emb_miss == 0,
            f"embarked missing={emb_miss}",
            f"embarked still missing={emb_miss}. Fill NaN with 'C'.")

        add("5) Imputation", "fare has no missing", 5,
            fare_miss == 0,
            f"fare missing={fare_miss}",
            f"fare still missing={fare_miss}. Fill NaN with median fare of pclass==3.")

        add("5) Imputation", "age has no missing", 6,
            age_miss == 0,
            f"age missing={age_miss}",
            f"age still missing={age_miss}. Use group median (sex,pclass), then fallback median.")

        # Check embarked NaNs set to C (if any in reference)
        mask_emb_ref = _REF["df"]["embarked"].isna()
        if int(mask_emb_ref.sum()) > 0:
            okC = (df_imp.loc[mask_emb_ref, "embarked"] == "C").all()
            add("5) Imputation", "missing embarked set to 'C'", 0,  # already rewarded above; keep as info only
                okC,
                "OK (missing embarked are 'C')",
                "Some missing embarked not set to 'C'.")
        else:
            # no points, informational
            add("5) Imputation", "missing embarked set to 'C' (ref had none)", 0,
                True,
                "Reference has no missing embarked.",
                "")
    else:
        add("5) Imputation", "embarked has no missing", 5, False, "OK.", "Cannot check without df_imp.")
        add("5) Imputation", "fare has no missing", 5, False, "OK.", "Cannot check without df_imp.")
        add("5) Imputation", "age has no missing", 6, False, "OK.", "Cannot check without df_imp.")

    # =================================================
    # Section 6: Build matrices (14)
    # =================================================
    Xa = ns.get("Xa", None)
    Xb = ns.get("Xb", None)
    y = ns.get("y", None)

    add("6) Matrices", "y is numpy array (shape matches)", 4,
        _is_array(y) and getattr(y, "shape", None) == _REF["y"].shape,
        f"y shape={_safe_shape(y)}",
        f"y missing/incorrect. Expected numpy shape {_REF['y'].shape}.")

    add("6) Matrices", "Xa is numpy array with 6 columns", 5,
        _is_array(Xa) and getattr(Xa, "shape", None) is not None and Xa.shape[1] == 6,
        f"Xa shape={_safe_shape(Xa)}",
        "Xa missing/incorrect. Expected numpy array with 6 columns: [sex, age, sibsp, parch, fare, embarked_num].")

    add("6) Matrices", "Xb is numpy array with 7 columns", 5,
        _is_array(Xb) and getattr(Xb, "shape", None) is not None and Xb.shape[1] == 7,
        f"Xb shape={_safe_shape(Xb)}",
        "Xb missing/incorrect. Expected numpy array with 7 columns: [sex, age, sibsp, parch, fare, emb_Q, emb_S].")

    # =================================================
    # Section 7: Modeling (18)
    # =================================================
    random_state = ns.get("random_state", None)
    add("7) Modeling", "random_state == 42", 3,
        _safe_float(random_state) is not None and _close(_safe_float(random_state), 42.0, tol=0.0),
        "random_state=42",
        "random_state missing/incorrect. Must be 42.")

    model_Xa = ns.get("model_Xa", None)
    model_Xb = ns.get("model_Xb", None)

    add("7) Modeling", "model_Xa is LogisticRegression", 3,
        isinstance(model_Xa, LogisticRegression),
        "model_Xa OK.",
        "model_Xa missing/incorrect. Must be LogisticRegression().")

    add("7) Modeling", "model_Xb is LogisticRegression", 3,
        isinstance(model_Xb, LogisticRegression),
        "model_Xb OK.",
        "model_Xb missing/incorrect. Must be LogisticRegression().")

    auc_Xa = _safe_float(ns.get("auc_Xa", None))
    auc_Xb = _safe_float(ns.get("auc_Xb", None))

    add("7) Modeling", "auc_Xa computed (plausible)", 4,
        auc_Xa is not None and 0.5 <= float(auc_Xa) <= 1.0,
        f"auc_Xa={_fmt(auc_Xa)}",
        "auc_Xa missing/incorrect. Compute ROC AUC on test split.")

    add("7) Modeling", "auc_Xb computed (plausible)", 5,
        auc_Xb is not None and 0.5 <= float(auc_Xb) <= 1.0,
        f"auc_Xb={_fmt(auc_Xb)}",
        "auc_Xb missing/incorrect. Compute ROC AUC on test split.")

    # Optional: closeness to reference (informational, 0 pts)
    if auc_Xa is not None:
        add("7) Modeling", "auc_Xa close to reference (info)", 0,
            _close(auc_Xa, _REF["auc_Xa"], tol=0.08),
            f"auc_Xa close (ref {_fmt(_REF['auc_Xa'])})",
            f"auc_Xa differs from ref (ref {_fmt(_REF['auc_Xa'])}).")
    if auc_Xb is not None:
        add("7) Modeling", "auc_Xb close to reference (info)", 0,
            _close(auc_Xb, _REF["auc_Xb"], tol=0.08),
            f"auc_Xb close (ref {_fmt(_REF['auc_Xb'])})",
            f"auc_Xb differs from ref (ref {_fmt(_REF['auc_Xb'])}).")

    # =================================================
    # Section 8: Required figures (6)
    # =================================================
    add("8) Outputs", "fig_roc_compare.png saved", 3,
        _file_exists("fig_roc_compare.png"),
        "Found fig_roc_compare.png",
        "Missing fig_roc_compare.png. Save ROC comparison plot with that filename.")

    add("8) Outputs", "fig_cm_Xb.png saved", 3,
        _file_exists("fig_cm_Xb.png"),
        "Found fig_cm_Xb.png",
        "Missing fig_cm_Xb.png. Save confusion matrix plot with that filename.")

    # =================================================
    # Section 9: Weights + feature names (6)
    # =================================================
    feat_names_Xa = ns.get("feat_names_Xa", None)
    feat_names_Xb = ns.get("feat_names_Xb", None)

    add("9) Interpretation", "feat_names_Xa length matches Xa", 3,
        isinstance(feat_names_Xa, list) and _is_array(Xa) and len(feat_names_Xa) == Xa.shape[1],
        f"len(feat_names_Xa)={len(feat_names_Xa) if isinstance(feat_names_Xa, list) else None}",
        "feat_names_Xa missing/incorrect. Must be list aligned with Xa columns.")

    add("9) Interpretation", "feat_names_Xb length matches Xb", 3,
        isinstance(feat_names_Xb, list) and _is_array(Xb) and len(feat_names_Xb) == Xb.shape[1],
        f"len(feat_names_Xb)={len(feat_names_Xb) if isinstance(feat_names_Xb, list) else None}",
        "feat_names_Xb missing/incorrect. Must be list aligned with Xb columns.")

    # -------------------------------------------------
    # Render
    # -------------------------------------------------
    total = sum(c.points for c in checks)
    # should be exactly 100; if changed, still display consistent totals
    _render_html(checks, score, total)
