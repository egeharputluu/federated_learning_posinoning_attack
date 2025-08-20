# client_1.py  (sadece: python client_1.py)
# data/client1.csv -> RandomForest eğitir
# models/client1_rf.pkl ve reports/client1_report.json üretir

from pathlib import Path
from typing import Optional
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CLIENT_ID = 1
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"

CSV_PATH = DATA_DIR / f"client{CLIENT_ID}.csv"
MODEL_OUT = MODELS_DIR / f"client{CLIENT_ID}_rf.pkl"
REPORT_OUT = REPORTS_DIR / f"client{CLIENT_ID}_report.json"

LABEL_CANDIDATES = ["label", "labels", "class", "target", "attack", "is_attack", "y"]

def detect_label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in LABEL_CANDIDATES or c.lower().startswith("label"):
            return c
    raise ValueError("Label (hedef) kolonu bulunamadı. (Örn: label/attack/target)")

def load_dataset(path: Path, label_col: Optional[str] = None):
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    if label_col is None:
        label_col = detect_label_column(df)

    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_enc, uniques = pd.factorize(y)
    return X_num, y_enc, uniques, label_col, len(df)

def train_local_rf(X, y, n_estimators=300, random_state=11, max_depth=None):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
    )
    clf.fit(X, y)
    return clf

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print(f"[Client{CLIENT_ID}] CSV: {CSV_PATH}")
    X, y, classes_, label_col, n_rows = load_dataset(CSV_PATH)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    clf = train_local_rf(Xtr, ytr, n_estimators=300, random_state=11)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    crep = classification_report(yte, ypred, zero_division=0, output_dict=True)

    joblib.dump({"model": clf, "classes_": classes_}, MODEL_OUT)

    report = {
        "client_id": CLIENT_ID,
        "data_path": str(CSV_PATH),
        "rows_total": int(n_rows),
        "label_column": label_col,
        "classes_mapping": {int(i): str(v) for i, v in enumerate(classes_)},
        "val_size": int(len(yte)),
        "accuracy": float(acc),
        "classification_report": crep,
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Client{CLIENT_ID}] Eğitim tamamlandı.")
    print(f"  Model : {MODEL_OUT}")
    print(f"  Rapor : {REPORT_OUT}")
    print(f"  Acc   : {acc:.4f}")

if __name__ == "__main__":
    main()


