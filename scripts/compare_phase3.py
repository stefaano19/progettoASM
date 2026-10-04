"""
scripts/compare_phase3.py
=========================
Effetto dell'intervento CELF = run con fact-checker  vs  run di controllo.

Le due run di Fase 3 devono partire dallo stesso checkpoint di fine Fase 2 ed
eseguire lo stesso numero di step. Il "prima/dopo" di una sola run mescola
l'effetto dei fact-checker con l'evoluzione spontanea della rete; il confronto
con il controllo isola l'effetto dell'intervento.

Uso
---
    python scripts/compare_phase3.py \\
        --treatment results/phase3_report.json \\
        --control   results/phase3_report_control.json \\
        [--csv results/metrics_history.csv --csv-control <altro csv>]

Scrive results/phase3_comparison.json e stampa una tabella.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = [
    ("Infection rate",       "infection_rate"),
    ("Nodi S",               "n_S"),
    ("Nodi I",               "n_I"),
    ("Nodi R",               "n_R"),
    ("Echo Chamber Index",   "echo_chamber_index"),
    ("Modularity Q",         "modularity_q"),
    ("Belief polarisation",  "belief_polarisation"),
    ("Opinion homophily",    "opinion_homophily"),
    ("Belief assortativity", "belief_assortativity"),
]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _phase3_series(csv_path: str, phase: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") == phase:
                rows.append(row)
    return rows


def compare(treat: dict, ctrl: dict) -> dict:
    out = {}
    for _, key in METRICS:
        t, c = treat.get(key), ctrl.get(key)
        if isinstance(t, (int, float)) and isinstance(c, (int, float)):
            out[key] = {"intervento": t, "controllo": c, "effetto": t - c}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treatment", default="results/phase3_report.json")
    ap.add_argument("--control", default="results/phase3_report_control.json")
    ap.add_argument("--csv", default=None, help="metrics_history.csv della run con intervento")
    ap.add_argument("--csv-control", default=None, help="metrics_history.csv della run di controllo")
    ap.add_argument("--out", default="results/phase3_comparison.json")
    args = ap.parse_args()

    treat, ctrl = _load(args.treatment), _load(args.control)
    if treat.get("control_run") or not ctrl.get("control_run", True):
        print("[Warn] Controlla i file: --treatment deve essere la run con fact-checker "
              "e --control quella di controllo.")
    if treat.get("n_post_steps") != ctrl.get("n_post_steps"):
        print(f"[Warn] Numero di step diverso: intervento={treat.get('n_post_steps')} "
              f"controllo={ctrl.get('n_post_steps')}. Il confronto non e' valido.")

    result = {"finale": compare(treat, ctrl)}

    print("=" * 72)
    print("EFFETTO DELL'INTERVENTO (intervento - controllo, a fine Fase 3)")
    print("=" * 72)
    print(f"{'Metrica':<24}{'Intervento':>14}{'Controllo':>14}{'Effetto':>14}")
    print("-" * 72)
    for label, key in METRICS:
        r = result["finale"].get(key)
        if r:
            print(f"{label:<24}{r['intervento']:>14.4f}{r['controllo']:>14.4f}{r['effetto']:>+14.4f}")
    print("=" * 72)

    if args.csv and args.csv_control:
        t_rows = _phase3_series(args.csv, "3")
        c_rows = _phase3_series(args.csv_control, "3_control")
        series = []
        for tr, cr in zip(t_rows, c_rows):
            series.append({
                "step": int(tr["step"]),
                "I_intervento": int(tr["I"]), "I_controllo": int(cr["I"]),
                "effetto_I": int(tr["I"]) - int(cr["I"]),
            })
        result["per_step"] = series
        if series:
            print(f"Effetto sugli infetti per step: "
                  f"{', '.join(str(s['effetto_I']) for s in series)}")

    print("\nNota: le risposte dell'LLM non sono perfettamente ripetibili; differenze "
          "piccole possono essere dovute al rumore.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Salvato: {args.out}")


if __name__ == "__main__":
    main()
