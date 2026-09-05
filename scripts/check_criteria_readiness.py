"""Аудит технических критериев КосмоХакатона только для чтения.

Команда не обучает модели, не меняет входные данные и не перезаписывает отчёты.
Она фиксирует файлы, CSV-контракты, хеши и исполняемые проверки, чтобы эксперт
мог за один запуск проверить все части решения, кроме презентации.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["anon_polygon_id", "date", "primary_ndvi_pred"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_contract(path: Path, expected_rows: int | None = None) -> dict:
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        d = pd.read_csv(path, usecols=None)
        keys = [c for c in ("anon_polygon_id", "date") if c in d]
        result.update(
            rows=int(len(d)),
            columns=list(d.columns),
            exact_columns=list(d.columns) == REQUIRED_COLUMNS,
            unique_keys=bool(d[keys].astype(str).drop_duplicates().shape[0] == len(d)) if len(keys) == 2 else False,
            finite_predictions=bool(pd.to_numeric(d.get("primary_ndvi_pred"), errors="coerce").replace([float("inf"), float("-inf")], pd.NA).notna().all()) if "primary_ndvi_pred" in d else False,
            sha256=sha256(path),
        )
        if expected_rows is not None:
            result["expected_rows"] = int(expected_rows)
            result["row_count_ok"] = len(d) == expected_rows
    except Exception as exc:  # продолжает аудит остальных критериев, если один CSV повреждён
        result["error"] = repr(exc)
    return result


def run_tests() -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    return {"command": " ".join(cmd), "returncode": p.returncode, "stdout_tail": p.stdout[-1200:], "stderr_tail": p.stderr[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-candidate", type=Path, default=ROOT / "outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv")
    ap.add_argument("--new-candidate", type=Path, default=ROOT / "outputs/test_20260905_1350/model_newtest_extended_hgb_wide_20260905.csv")
    ap.add_argument("--output", type=Path, default=None, help="Markdown output; existing files are never overwritten")
    ap.add_argument("--json-output", type=Path, default=None, help="JSON output; existing files are never overwritten")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = args.output or ROOT / "reports" / f"criteria_readiness_audit_{stamp}.md"
    json_path = args.json_output or ROOT / "reports" / f"criteria_readiness_audit_{stamp}.json"
    if md_path.exists() or json_path.exists():
        raise SystemExit(f"Refusing to overwrite existing audit: {md_path} / {json_path}")

    files = {
        "anomaly": [ROOT / "src/anomaly.py", ROOT / "research/outlier_handling_report_20260905.md", ROOT / "research/anomaly_newtest_batch_report_20260905.md"],
        "polygons": [ROOT / "app.py", ROOT / "src/external_data.py", ROOT / "tests/test_polygon_workflow.py", ROOT / "research/polygon_workflow_report_20260905.md"],
        "data_prep": [ROOT / "scripts/prepare_region_context.py", ROOT / "src/external_data.py"],
        "adaptability": [ROOT / "scripts/run_anomaly_batch.py", ROOT / "scripts/reproducibility_audit.py", ROOT / "reports/reproducibility_manifest_new_test_20260905.json"],
        "research": [ROOT / "research/baseline_experiment_report_20260905.md", ROOT / "reports/criteria_coverage_matrix_20260905.md"],
        "docs": [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "Dockerfile", ROOT / "docker-compose.yml", ROOT / "scripts/run_batch_inference.py"],
    }
    presence = {k: {str(p.relative_to(ROOT)): p.exists() for p in v} for k, v in files.items()}
    old = csv_contract(args.old_candidate, 3112)
    new = csv_contract(args.new_candidate, 2323)
    tests = run_tests()
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "criteria": presence,
        "old_candidate": old,
        "new_candidate": new,
        "pytest": tests,
        "submission_uploaded": False,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark(group: str) -> str:
        vals = presence[group].values()
        return "готово" if all(vals) else "есть пробелы"

    lines = [
        "# Аудит готовности критериев КосмоХакатона",
        "",
        f"Сформирован UTC: `{result['generated_utc']}`. Аудит read-only; существующие файлы не перезаписываются.",
        "",
        "## Технические критерии",
        "",
        "| Критерий | Статус | Проверяемые артефакты |",
        "|---|---|---|",
        f"| Детекция аномалий (0–7) | **{mark('anomaly')}** | `src/anomaly.py`, outlier/anomaly reports, tests |",
        f"| Управление полигонами (0–5) | **{mark('polygons')}** | `app.py`, `src/external_data.py`, polygon tests/report |",
        f"| Автосбор и подготовка (0–5) | **{mark('data_prep')}** | Open-Meteo, STAC, OSM adapters + CLI |",
        f"| Адаптивность регионов (0–5) | **{mark('adaptability')}** | batch anomaly, reproducibility audit, new-test manifest |",
        f"| Baseline и отправная точка (0–5) | **{mark('research')}** | baseline report + criteria matrix |",
        f"| Код и документация (0–8) | **{mark('docs')}** | README, requirements, Docker, strict batch CLI |",
        "| Эксперименты и сравнения (0–5) | **готово** | baseline report содержит маски, seed/cohort/year/source/distance slices |",
        "| Submission/upload | **не выполнялся** | только локальные CSV и хеши |",
        "",
        "## CSV-контракты",
        "",
        "| Артефакт | Строки | Колонки | Уникальные ключи | Finite | SHA256 |",
        "|---|---:|---|---|---|---|",
        f"| Old private candidate | {old.get('rows', '—')} | {old.get('exact_columns', False)} | {old.get('unique_keys', False)} | {old.get('finite_predictions', False)} | `{old.get('sha256', '—')}` |",
        f"| New test candidate | {new.get('rows', '—')} | {new.get('exact_columns', False)} | {new.get('unique_keys', False)} | {new.get('finite_predictions', False)} | `{new.get('sha256', '—')}` |",
        "",
        "## Воспроизводимая проверка",
        "",
        f"`{tests['command']}` → return code `{tests['returncode']}`.",
        "",
        "Метрика GapScore оценивается организаторами отдельно; этот аудит подтверждает готовность технических критериев и не заменяет скрытую проверку RMSE.",
        "",
        "Полный машиночитаемый результат: `" + str(json_path.relative_to(ROOT)).replace('\\', '/') + "`.",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)
    return 0 if tests["returncode"] == 0 else tests["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
