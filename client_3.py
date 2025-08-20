# # client_3.py  (sadece: python client_3.py)
#
# from pathlib import Path
# from typing import Optional
# import json
#
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
#
# CLIENT_ID = 3
# DATA_DIR = Path(__file__).parent / "data"
# MODELS_DIR = Path(__file__).parent / "models"
# REPORTS_DIR = Path(__file__).parent / "reports"
#
# CSV_PATH = DATA_DIR / f"client{CLIENT_ID}.csv"
# MODEL_OUT = MODELS_DIR / f"client{CLIENT_ID}_rf.pkl"
# REPORT_OUT = REPORTS_DIR / f"client{CLIENT_ID}_report.json"
#
# LABEL_CANDIDATES = ["label", "labels", "class", "target", "attack", "is_attack", "y"]
#
# def detect_label_column(df: pd.DataFrame) -> str:
#     for c in df.columns:
#         if c.lower() in LABEL_CANDIDATES or c.lower().startswith("label"):
#             return c
#     raise ValueError("Label (hedef) kolonu bulunamadı.")
#
# def load_dataset(path: Path, label_col: Optional[str] = None):
#     if not path.exists():
#         raise FileNotFoundError(f"CSV bulunamadı: {path}")
#     try:
#         df = pd.read_csv(path, low_memory=False)
#     except UnicodeDecodeError:
#         df = pd.read_csv(path, low_memory=False, encoding="latin-1")
#
#     if label_col is None:
#         label_col = detect_label_column(df)
#
#     X = df.drop(columns=[label_col])
#     y = df[label_col]
#     X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
#     y_enc, uniques = pd.factorize(y)
#     return X_num, y_enc, uniques, label_col, len(df)
#
# def train_local_rf(X, y, n_estimators=300, random_state=33, max_depth=None):
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         n_jobs=-1,
#         max_depth=max_depth,
#     )
#     clf.fit(X, y)
#     return clf
#
# def main():
#     MODELS_DIR.mkdir(exist_ok=True)
#     REPORTS_DIR.mkdir(exist_ok=True)
#
#     print(f"[Client{CLIENT_ID}] CSV: {CSV_PATH}")
#     X, y, classes_, label_col, n_rows = load_dataset(CSV_PATH)
#
#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
#
#     clf = train_local_rf(Xtr, ytr, n_estimators=300, random_state=33)
#     ypred = clf.predict(Xte)
#     acc = accuracy_score(yte, ypred)
#     crep = classification_report(yte, ypred, zero_division=0, output_dict=True)
#
#     joblib.dump({"model": clf, "classes_": classes_}, MODEL_OUT)
#
#     report = {
#         "client_id": CLIENT_ID,
#         "data_path": str(CSV_PATH),
#         "rows_total": int(n_rows),
#         "label_column": label_col,
#         "classes_mapping": {int(i): str(v) for i, v in enumerate(classes_)},
#         "val_size": int(len(yte)),
#         "accuracy": float(acc),
#         "classification_report": crep,
#     }
#     with open(REPORT_OUT, "w", encoding="utf-8") as f:
#         json.dump(report, f, ensure_ascii=False, indent=2)
#
#     print(f"[Client{CLIENT_ID}] Eğitim tamamlandı.")
#     print(f"  Model : {MODEL_OUT}")
#     print(f"  Rapor : {REPORT_OUT}")
#     print(f"  Acc   : {acc:.4f}")
#
# if __name__ == "__main__":
#     main()




































# client_3.py
from pathlib import Path
from typing import Optional
import json
import os
import threading
import socket
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CLIENT_ID = 3
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"

CSV_PATH = DATA_DIR / f"client{CLIENT_ID}.csv"
MODEL_OUT = MODELS_DIR / f"client{CLIENT_ID}_rf.pkl"
REPORT_OUT = REPORTS_DIR / f"client{CLIENT_ID}_report.json"

LABEL_CANDIDATES = ["label", "labels", "class", "target", "attack", "is_attack", "y"]

# ---------- Opsiyonel: Basit TCP server (Nmap görünürlüğü için) ----------
# OpenSSH zaten 22/tcp'de çalışıyorsa buna gerçekten ihtiyacın yok.
# Yine de istersen PORT 2222'de basit bir banner servisi açabilirsin:
START_FAKE_TCP = os.getenv("START_FAKE_TCP", "0")   # "1" yaparsan başlar
FAKE_TCP_HOST = os.getenv("FAKE_TCP_HOST", "0.0.0.0")
FAKE_TCP_PORT = int(os.getenv("FAKE_TCP_PORT", "2222"))
FAKE_BANNER = b"SSH-2.0-OpenSSH_8.9p1\r\n"  # Nmap -sV için SSH benzeri banner

def detect_label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in LABEL_CANDIDATES or c.lower().startswith("label"):
            return c
    raise ValueError("Label (hedef) kolonu bulunamadı.")

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

def train_local_rf(X, y, n_estimators=300, random_state=33, max_depth=None):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
    )
    clf.fit(X, y)
    return clf

def fake_tcp_server(host: str, port: int):
    """Nmap -sV için banner veren basit TCP servis (gerçek SSH değildir)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(50)
    print(f"[Client{CLIENT_ID}] Fake TCP server listening on {host}:{port}")
    try:
        while True:
            conn, addr = srv.accept()
            try:
                conn.sendall(FAKE_BANNER)
                time.sleep(0.05)
            finally:
                conn.close()
    except Exception as e:
        print(f"[Client{CLIENT_ID}] Fake TCP server error: {e}")
    finally:
        srv.close()

def maybe_start_fake_tcp():
    if START_FAKE_TCP == "1":
        t = threading.Thread(target=fake_tcp_server, args=(FAKE_TCP_HOST, FAKE_TCP_PORT), daemon=True)
        t.start()
        time.sleep(0.2)

def check_ssh_open(host="127.0.0.1", port=22, timeout=2.0):
    """OpenSSH servisinin ayakta olup olmadığını hızlıca kontrol eder."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            # SSH sunucuları genellikle ilk paket olarak banner yollar.
            try:
                banner = s.recv(128)
            except Exception:
                banner = b""
            print(f"[Client{CLIENT_ID}] SSH check: {host}:{port} OPEN. Banner={banner!r}")
            return True
    except Exception as e:
        print(f"[Client{CLIENT_ID}] SSH check FAILED: {host}:{port} ({e})")
        return False

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print(f"[Client{CLIENT_ID}] CSV: {CSV_PATH}")
    X, y, classes_, label_col, n_rows = load_dataset(CSV_PATH)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    clf = train_local_rf(Xtr, ytr, n_estimators=300, random_state=33)
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

    # --- OpenSSH kontrolü (22/tcp açık mı?) ---
    check_ssh_open(host=os.getenv("SSH_CHECK_HOST", "127.0.0.1"),
                   port=int(os.getenv("SSH_CHECK_PORT", "22")))

    # --- Opsiyonel: Fake TCP server (gerekirse) ---
    maybe_start_fake_tcp()
    if START_FAKE_TCP == "1":
        print(f"[Client{CLIENT_ID}] Nmap örnek: nmap -sV -p {FAKE_TCP_PORT} 127.0.0.1")

    print(f"[Client{CLIENT_ID}] Brute force testin için hazır:")
    print("  Paramiko örnek: port=22, host=127.0.0.1 (OpenSSH)")
    print("  Nmap örnek    : nmap -sV -p 22 127.0.0.1")

    # Process yaşamda kalsın ki port takibi/log yapılabilsin
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print(f"[Client{CLIENT_ID}] Kapatılıyor...")

if __name__ == "__main__":
    main()

