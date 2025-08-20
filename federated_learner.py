# federated_learner.py — Client accuracy ortalaması ile federated skor

from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"

CLIENT_REPORT_FILES = [REPORTS_DIR / f"client{i}_report.json" for i in (1, 2, 3)]
OUT_JSON = REPORTS_DIR / "federated_report.json"

def main():
    # Client raporlarını oku
    accuracies = []
    client_infos = []
    for rep_path in CLIENT_REPORT_FILES:
        if not rep_path.exists():
            raise FileNotFoundError(f"Rapor yok: {rep_path}")

        with open(rep_path, "r", encoding="utf-8") as f:
            rep = json.load(f)

        acc = float(rep.get("accuracy", 0.0))
        accuracies.append(acc)
        client_infos.append({
            "client_id": rep.get("client_id"),
            "data_path": rep.get("data_path"),
            "rows_total": rep.get("rows_total"),
            "label_column": rep.get("label_column"),
            "accuracy": acc
        })

    # Federated skor = client accuracy ortalaması
    mean_acc = float(np.mean(accuracies))

    # Raporu kaydet
    report = {
        "aggregation": "mean_client_accuracy",
        "n_clients": len(CLIENT_REPORT_FILES),
        "client_details": client_infos,
        "federated_score": mean_acc
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Konsola yaz
    print("[Federated] Tamamlandı.")
    print(f"  Toplama yöntemi   : {report['aggregation']}")
    print(f"  Client accuracies : {', '.join(f'{a:.4f}' for a in accuracies)}")
    print(f"  Federated skor    : {mean_acc:.4f}")
    print(f"  Rapor JSON        : {OUT_JSON}")

if __name__ == "__main__":
    main()
