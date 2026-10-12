"""
train_model.py  -  Diabetes risk model (Pima Indians Diabetes dataset)

Run:   python train_model.py
Needs: pima.csv in the same folder (or in ./data/). Works with or without a header row.

Outputs:
  model.joblib   best model + decision threshold (for a FastAPI backend)
  model.json     logistic-regression weights (for the Streamlit app and index.html)
  roc_curve.png  ROC curve of the best model (if matplotlib is installed)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SEED = 42
COLS = ["pregnancies", "glucose", "bloodPressure", "skinThickness",
        "insulin", "bmi", "pedigree", "age", "outcome"]
ZERO_IS_MISSING = ["glucose", "bloodPressure", "skinThickness", "insulin", "bmi"]


def load_data() -> pd.DataFrame:
    path = next((p for p in (HERE / "pima.csv", HERE / "data" / "pima.csv", Path("pima.csv"))
                 if p.exists()), None)
    if path is None:
        raise FileNotFoundError("pima.csv not found. Put it next to train_model.py or in ./data/")
    df = pd.read_csv(path, header=None)
    # If the first row is a text header (e.g. "Pregnancies,Glucose,..."), re-read it as a header
    if not pd.to_numeric(df.iloc[0], errors="coerce").notna().all():
        df = pd.read_csv(path, header=0)
    df = df.iloc[:, :9]
    df.columns = COLS
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["outcome"])
    df["outcome"] = df["outcome"].astype(int)
    for c in ZERO_IS_MISSING:           # a zero here means "not measured"
        df[c] = df[c].replace(0, np.nan)
    print(f"Loaded {len(df)} rows from {path} | diabetic: {df.outcome.mean():.1%}")
    return df


def build_models():
    imp = lambda: SimpleImputer(strategy="median")
    models = {
        "Logistic Regression": GridSearchCV(
            make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=2000)),
            {"logisticregression__C": [0.01, 0.1, 0.5, 1, 5, 10]}, cv=5, scoring="roc_auc"),
        "Random Forest": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=3, random_state=SEED)),
        "Gradient Boosting": make_pipeline(imp(), GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = make_pipeline(imp(), XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
            eval_metric="logloss", random_state=SEED))
    except ImportError:
        print("(xgboost not installed, skipping it)")
    return models


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(accuracy=round(accuracy_score(y, pred), 3), auc=round(roc_auc_score(y, proba), 3),
                precision=round(precision_score(y, pred), 3), recall=round(recall_score(y, pred), 3),
                f1=round(f1_score(y, pred), 3))


def main():
    df = load_data()
    X, y = df.drop(columns="outcome"), df["outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    report, fitted = {}, {}
    print("\nModel comparison (test set = 20% unseen data)")
    for name, model in build_models().items():
        cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        model.fit(Xtr, ytr)
        m = metrics(yte, model.predict_proba(Xte)[:, 1])
        m["cvAuc"] = round(float(cv_auc), 3)
        report[name], fitted[name] = m, model
        print(f"  {name:<20} CV AUC {m['cvAuc']:.3f} | acc {m['accuracy']:.3f} | "
              f"AUC {m['auc']:.3f} | recall {m['recall']:.3f} | precision {m['precision']:.3f}")

    best = max(report, key=lambda n: report[n]["cvAuc"])
    print(f"\nBest model by cross-validated AUC: {best}")

    # Screening threshold: maximise F2 (recall counts 2x) using out-of-fold predictions on train data
    oof = cross_val_predict(fitted[best], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(ytr, oof)
    f2 = 5 * prec[:-1] * rec[:-1] / np.maximum(4 * prec[:-1] + rec[:-1], 1e-9)
    threshold = float(thr[np.argmax(f2)])
    final = metrics(yte, fitted[best].predict_proba(Xte)[:, 1], threshold)
    print(f"Screening threshold {threshold:.2f} -> test recall {final['recall']:.3f}, "
          f"precision {final['precision']:.3f}, accuracy {final['accuracy']:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n",
          confusion_matrix(yte, (fitted[best].predict_proba(Xte)[:, 1] >= threshold).astype(int)))

    # Realistic accuracy estimate: 10-fold CV on the full dataset
    acc10 = cross_val_score(fitted[best], X, y, cv=StratifiedKFold(10, shuffle=True, random_state=1),
                            scoring="accuracy").mean()
    print(f"10-fold CV accuracy on all data: {acc10:.3f}")

    joblib.dump({"model": fitted[best], "threshold": threshold, "features": list(X.columns),
                 "name": best}, HERE / "model.joblib")

    # Export logistic regression weights for the Streamlit app / index.html
    lr = fitted["Logistic Regression"].best_estimator_
    imp, sc, clf = lr.steps[0][1], lr.steps[1][1], lr.steps[2][1]
    json.dump({"features": list(X.columns), "median": imp.statistics_.tolist(),
               "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
               "cvAccuracy10": round(float(acc10), 3), "report": report},
              open(HERE / "model.json", "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_estimator(fitted[best], Xte, yte)
        plt.title(f"ROC curve: {best}")
        plt.savefig(HERE / "roc_curve.png", dpi=150, bbox_inches="tight")
    except ImportError:
        pass
    print("\nSaved: model.joblib, model.json" + (", roc_curve.png" if (HERE / 'roc_curve.png').exists() else ""))


if __name__ == "__main__":
    main()