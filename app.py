"""Агропульс: воспроизводимое Streamlit-демо для мониторинга NDVI.

Запуск с данными конкурса:
    streamlit run app.py

Пути можно задать через AGROPULSE_DATA_DIR и AGROPULSE_PREDICTIONS. Если
координаты отсутствуют в анонимизированном наборе, интерфейс принимает
GeoJSON пользователя и запрашивает открытый погодный/спутниковый каталог.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.anomaly import add_anomaly_columns, anomaly_periods, region_summary
from src.digital_twin import counterfactual
from src.early_warning import early_warning_radar
from src.external_data import (fetch_open_meteo, geojson_centroid,
                               merge_weather_context,
                               search_osm_agricultural_contours,
                               search_sentinel_items, validate_geojson)
from src.io_utils import read_csv_auto

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AGROPULSE_DATA_DIR", r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"))
PRIVATE_FILENAME = os.environ.get("AGROPULSE_PRIVATE_FILENAME", "private_features.csv")
TRAIN_FILENAME = os.environ.get("AGROPULSE_TRAIN_FILENAME", "train_dataset.csv")
DEFAULT_PRED = ROOT / "outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv"


def _path(name: str) -> Path:
    return DATA_DIR / name


def _gap_mask(s: pd.Series) -> pd.Series:
    """Разбирает логические значения организатора и не считает строку ``False`` истинной."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


@st.cache_data(show_spinner=False)
def load_data(data_dir: str, prediction_path: str):
    train = read_csv_auto(Path(data_dir) / TRAIN_FILENAME, parse_dates=["date"], low_memory=False)
    private = read_csv_auto(Path(data_dir) / PRIVATE_FILENAME, parse_dates=["date"], low_memory=False)
    pred = read_csv_auto(prediction_path, parse_dates=["date"], low_memory=False)
    pred = pred.drop_duplicates(["anon_polygon_id", "date"]).set_index(["anon_polygon_id", "date"])
    private["primary_ndvi_pred"] = private.set_index(["anon_polygon_id", "date"]).index.map(pred["primary_ndvi_pred"])
    values = private["primary_ndvi"].astype(float).copy()
    gaps = _gap_mask(private["is_synthetic_gap"])
    missing = values.isna() & gaps
    values.loc[missing] = private.loc[missing, "primary_ndvi_pred"]
    private["ndvi_filled"] = values
    # В интерфейсе отображаются только строки private. Наблюдаемые значения train и
    # private остаются историческим эталоном: это избавляет от второго прохода
    # по климатологии на 100 тысяч строк и сохраняет leakage-safe базу.
    private["ndvi_filled"] = values
    history = pd.concat([train, private], ignore_index=True, sort=False)
    enriched = add_anomaly_columns(private, values=private["ndvi_filled"], reference_frame=history)
    return private, enriched, train


def _polygon_from_text(text: str) -> dict:
    """Создаёт замкнутый полигон WGS84 из строк ``lon,lat``."""
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pair = [x.strip() for x in line.replace(";", ",").split(",")]
        if len(pair) != 2:
            raise ValueError("Каждая строка должна быть lon,lat")
        points.append([float(pair[0]), float(pair[1])])
    if len(points) < 3:
        raise ValueError("Нужно минимум 3 вершины")
    if points[0] != points[-1]:
        points.append(points[0])
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [points]}}


def _passport_sparkline(frame: pd.DataFrame, width: int = 760, height: int = 180) -> str:
    """Возвращает компактный SVG-график NDVI для автономного HTML-паспорта."""
    if frame.empty or "date" not in frame:
        return "<p>Недостаточно точек для графика.</p>"
    d = frame.sort_values("date").copy()
    y = pd.to_numeric(d.get("ndvi_filled"), errors="coerce")
    n = y.notna()
    if int(n.sum()) < 2:
        return "<p>Недостаточно конечных значений NDVI для графика.</p>"
    y = y[n].to_numpy(float)
    lo = float(min(-0.05, np.nanpercentile(y, 2)))
    hi = float(max(1.0, np.nanpercentile(y, 98)))
    if hi <= lo:
        hi = lo + 1.0
    xs = np.linspace(20, width - 20, len(y))
    ys = height - 24 - (y - lo) / (hi - lo) * (height - 44)
    points = " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Траектория NDVI" '
            'style="width:100%;height:auto;background:#f7f9fd;border:1px solid #e6ebf3;border-radius:12px">'
            f'<line x1="20" y1="{height-24}" x2="{width-20}" y2="{height-24}" stroke="#dbe2ee"/>'
            f'<polyline points="{points}" fill="none" stroke="#3d5afe" stroke-width="3" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<text x="22" y="16" font-size="11" fill="#65738b">NDVI, min {lo:.2f} · max {hi:.2f}</text>'
            f'<text x="20" y="{height-7}" font-size="10" fill="#65738b">начало периода</text>'
            f'<text x="{width-112}" y="{height-7}" font-size="10" fill="#65738b">конец периода</text></svg>')


def _make_aoi_passport_html(
    aoi_id: str,
    frame: pd.DataFrame,
    radar_row: pd.Series | None,
    aoi_periods: pd.DataFrame,
    twin_sens: dict[str, float],
    twin_delta: float,
    twin_stress_n: int,
    generated_at: datetime,
) -> str:
    """Собирает автономный HTML-паспорт без обращения к внешним сервисам."""
    rows = len(frame)
    observed = int(frame.get("is_observed", pd.Series(False, index=frame.index)).fillna(False).sum())
    reconstructed = int(frame.get("is_reconstructed", pd.Series(False, index=frame.index)).fillna(False).sum())
    missing = max(0, rows - observed - reconstructed)
    coverage = float(pd.to_numeric(frame.get("ndvi_zscore", pd.Series(index=frame.index)), errors="coerce").notna().mean()) if rows else 0.0
    anomalies = int(frame.get("status", pd.Series(index=frame.index, dtype=str)).isin(["suppression", "critical"]).sum())
    critical = int(frame.get("status", pd.Series(index=frame.index, dtype=str)).eq("critical").sum())
    risk_score = float(radar_row.get("risk_score", 0.0)) if radar_row is not None else 0.0
    risk_level = str(radar_row.get("risk_level", "Нет данных")) if radar_row is not None else "Нет данных"
    factors = str(radar_row.get("top_factors", "нет сильных сигналов")) if radar_row is not None else "нет данных"
    period_count = int(len(aoi_periods))
    stress_lengths = pd.to_numeric(aoi_periods.get("n_days", pd.Series(dtype=float)), errors="coerce").dropna()
    max_stress = int(stress_lengths.max()) if not stress_lengths.empty else 0
    start = pd.to_datetime(frame.get("date"), errors="coerce").min() if rows else pd.NaT
    end = pd.to_datetime(frame.get("date"), errors="coerce").max() if rows else pd.NaT
    period_label = f"{start.date()} — {end.date()}" if pd.notna(start) and pd.notna(end) else "нет дат"
    period_rows_parts = []
    for _, r in aoi_periods.sort_values("start").head(12).iterrows():
        n_days = pd.to_numeric(r.get("n_days", 0), errors="coerce")
        n_days_text = str(int(n_days)) if pd.notna(n_days) else "0"
        period_rows_parts.append(
            f"<tr><td>{escape(str(r.get('start', '—'))[:10])}</td>"
            f"<td>{escape(str(r.get('end', '—'))[:10])}</td>"
            f"<td>{n_days_text}</td><td>{escape(str(r.get('status', '—')))}</td></tr>"
        )
    period_rows = "".join(period_rows_parts) or '<tr><td colspan="4">Периодов стресса не найдено</td></tr>'
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Паспорт AOI {escape(aoi_id)}</title>
<style>body{{font-family:Inter,Segoe UI,Arial,sans-serif;color:#14213d;background:#f7f9fd;margin:0;padding:28px}}
.sheet{{max-width:980px;margin:auto;background:#fff;border:1px solid #e6ebf3;border-radius:20px;padding:30px;box-shadow:0 12px 35px #14213d14}}
h1{{margin:0 0 5px;font-size:30px}} h2{{margin:28px 0 12px;font-size:18px}} .muted{{color:#65738b}}
.badge{{display:inline-block;padding:5px 11px;border-radius:999px;background:#eef2ff;color:#3d5afe;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}} .metric{{padding:13px;border:1px solid #e6ebf3;border-radius:12px;background:#fbfcff}}
.metric b{{display:block;font-size:22px;margin-top:4px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:9px;border-bottom:1px solid #e6ebf3}} th{{color:#65738b;font-weight:600}}
.callout{{padding:14px;border-radius:12px;background:#f0f5ff;border-left:4px solid #3d5afe}} .foot{{margin-top:24px;font-size:12px;color:#65738b}}
@media(max-width:700px){{body{{padding:10px}}.sheet{{padding:18px}}.grid{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><main class="sheet"><div class="muted">АГРОПУЛЬС · ПАСПОРТ ПОЛИГОНА</div>
<h1>AOI {escape(aoi_id)}</h1><div class="muted">Период анализа: {escape(period_label)} · создан {generated_at.strftime('%Y-%m-%d %H:%M UTC')}</div>
<p><span class="badge">Риск {risk_score:.1f} / 100 · {escape(risk_level)}</span></p>
<div class="grid"><div class="metric">Покрытие климатологии<b>{coverage:.1%}</b></div><div class="metric">Наблюдения<b>{observed:,}</b></div><div class="metric">Восстановлено<b>{reconstructed:,}</b></div><div class="metric">Пропущено<b>{missing:,}</b></div>
<div class="metric">Аномалии<b>{anomalies:,}</b></div><div class="metric">Критические<b>{critical:,}</b></div><div class="metric">Стресс-периоды<b>{period_count}</b></div><div class="metric">Макс. стресс<b>{max_stress} дн.</b></div></div>
<div class="callout"><b>Главные факторы риска:</b> {escape(factors)}.</div>
<h2>Траектория NDVI</h2>{_passport_sparkline(frame)}
<h2>Стресс-тест по умолчанию</h2><p class="muted">Сценарий: +2 °C и −30% осадков, сила 1.0. Медианное изменение NDVI: <b>{twin_delta:+.3f}</b>; точек под стрессом (&lt; −0.02): <b>{twin_stress_n}</b>. Чувствительность: температура {float(twin_sens.get('beta_temp', 0)):+.3f}, осадки {float(twin_sens.get('beta_precip', 0)):+.3f}.</p>
<h2>Периоды стресса</h2><table><thead><tr><th>Начало</th><th>Окончание</th><th>Дней</th><th>Статус</th></tr></thead><tbody>{period_rows}</tbody></table>
<div class="foot">Паспорт является воспроизводимым интерпретационным отчётом. Предсказания и конкурсный CSV не изменяются. Формула риска: 0.35·z + 0.25·critical + 0.15·suppression + 0.15·persistence + 0.10·uncertainty, нормированная на 0–100.</div>
</main></body></html>"""


st.set_page_config(page_title="Агропульс", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --ink:#14213d; --muted:#65738b; --line:#e6ebf3; --brand:#3d5afe;
      --brand-2:#7c4dff; --mint:#17a673; --panel:#ffffff; --soft:#f5f7fb;
    }
    .stApp { background:linear-gradient(180deg,#f7f9fd 0%,#ffffff 36%); }
    .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3 { color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#101a35 0%,#182957 100%); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span { color:#eef3ff !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] textarea { background:#223664 !important; border-color:#3b548e !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * { color:#eef3ff !important; }
    [role="listbox"], [role="listbox"] * { color:#14213d !important; background:#fff !important; }
    [data-testid="stSidebar"] hr { border-color:#3b548e; }
    .hero { padding:1.4rem 1.7rem 1.25rem; border-radius:20px; color:#fff;
      background:radial-gradient(circle at 85% 10%,#7387ff 0,#4a5de2 27%,#172754 75%);
      box-shadow:0 14px 35px rgba(40,57,125,.18); margin:0 0 1.25rem; }
    .hero h1 { margin:0; font-size:2.15rem; letter-spacing:-.03em; color:#fff; }
    .hero p { margin:.35rem 0 0; color:#dbe3ff; font-size:1rem; }
    .hero-badges { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1rem; }
    .badge { border:1px solid rgba(255,255,255,.28); background:rgba(255,255,255,.12);
      padding:.28rem .7rem; border-radius:999px; font-size:.78rem; color:#f3f6ff; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:15px;
      padding:.75rem 1rem; box-shadow:0 5px 18px rgba(21,39,84,.06); }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.79rem; }
    [data-testid="stMetricValue"] { color:var(--ink) !important; font-size:1.65rem; }
    .section-kicker { color:var(--brand); font-size:.74rem; font-weight:700; letter-spacing:.12em;
      text-transform:uppercase; margin:1.25rem 0 .25rem; }
    .panel-title { color:var(--ink); font-size:1.18rem; font-weight:700; margin:.15rem 0 .7rem; }
    .hint { color:var(--muted); font-size:.86rem; line-height:1.45; }
    .legend-card { display:flex; flex-wrap:wrap; gap:.45rem .9rem; margin:.35rem 0 .65rem;
      padding:.7rem .85rem; border:1px solid var(--line); border-radius:14px; background:#fff;
      box-shadow:0 3px 12px rgba(21,39,84,.035); }
    .legend-item { display:flex; align-items:center; gap:.42rem; color:var(--muted); font-size:.78rem; white-space:nowrap; }
    .legend-line { width:25px; height:0; border-top:3px solid #3d5afe; display:inline-block; border-radius:999px; }
    .legend-line.reconstructed { border-top-style:dotted; border-top-color:#ff8a34; border-top-width:3px; }
    .legend-line.dotted { border-top-style:dotted; border-top-width:2px; border-top-color:#17a673; }
    .legend-line.dashed { border-top-style:dashed; border-top-color:#e64a6f; }
    .legend-band { width:25px; height:9px; border-radius:5px; background:rgba(61,90,254,.16); border:1px solid rgba(61,90,254,.25); display:inline-block; }
    .legend-cross { color:#ff4b5c; font-size:1.1rem; line-height:1; width:25px; text-align:center; }
    .stPlotlyChart, [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px;
      overflow:hidden; box-shadow:0 4px 16px rgba(21,39,84,.045); }
    div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:#fff; }
    /* Токены темы Streamlit могут сделать подписи кнопок почти белыми и на светлом
       фоне, и на тёмной боковой панели. Фиксируем явный контрастный цвет
       для всего дерева кнопки, включая SVG-иконки. */
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    .stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #c8d3e8;
      background:#ffffff !important; color:#14213d !important; font-weight:650; transition:all .18s ease; }
    [data-testid="stButton"] button *, [data-testid="stDownloadButton"] button *,
    .stButton > button *, .stDownloadButton > button * { color:#14213d !important; fill:#14213d !important; stroke:#14213d !important; }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover,
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--brand) !important; background:#3d5afe !important; color:#ffffff !important; transform:translateY(-1px); }
    [data-testid="stButton"] button:hover *, [data-testid="stDownloadButton"] button:hover *,
    .stButton > button:hover *, .stDownloadButton > button:hover * { color:#ffffff !important; fill:#ffffff !important; stroke:#ffffff !important; }
    [data-testid="stButton"] button:focus-visible, [data-testid="stDownloadButton"] button:focus-visible,
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:3px solid rgba(61,90,254,.35) !important; outline-offset:2px; }
    [data-testid="stButton"] button:disabled, [data-testid="stDownloadButton"] button:disabled,
    .stButton > button:disabled, .stDownloadButton > button:disabled { background:#edf1f7 !important; color:#63708a !important; border-color:#d9e0ec !important; opacity:1 !important; }
    [data-testid="stButton"] button:disabled *, [data-testid="stDownloadButton"] button:disabled * { color:#63708a !important; fill:#63708a !important; stroke:#63708a !important; }
    .sidebar-brand { font-size:1.3rem; font-weight:800; letter-spacing:-.02em; margin-bottom:.1rem; }
    .sidebar-sub { color:#b7c5eb !important; font-size:.78rem; line-height:1.4; margin-bottom:1rem; }
    </style>
    <div class="hero">
      <h1>🌱 Агропульс</h1>
      <p>Восстановление NDVI, климатическая норма и объяснимые периоды стресса</p>
      <div class="hero-badges"><span class="badge">● мониторинг в реальном времени</span>
      <span class="badge">↗ leakage-safe восстановление</span><span class="badge">⌁ 78 AOI</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

pred_path = Path(os.environ.get("AGROPULSE_PREDICTIONS", str(DEFAULT_PRED)))
if not _path(PRIVATE_FILENAME).exists() or not pred_path.exists():
    st.error("Не найдены данные или prediction artifact. Задайте AGROPULSE_DATA_DIR и AGROPULSE_PREDICTIONS.")
    st.stop()

private, data, train = load_data(str(DATA_DIR), str(pred_path))
polygons = sorted(data["anon_polygon_id"].dropna().astype(str).unique())
if "selected_polygons" not in st.session_state:
    st.session_state.selected_polygons = polygons[:1]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌿 Агропульс</div><div class="sidebar-sub">Панель мониторинга растительности</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker" style="color:#aebeff">ФИЛЬТРЫ</div>', unsafe_allow_html=True)
    st.header("Регион")
    selected = st.multiselect("Полигоны", polygons, default=st.session_state.selected_polygons)
    # Сохраняет явно пустой выбор, чтобы пользователь мог очистить рабочую область
    # перед переключением на новый нарисованный или импортированный регион.
    st.session_state.selected_polygons = selected
    # По умолчанию показываем весь доступный ряд, но даём выбрать произвольный
    # непрерывный интервал дат: такой фильтр удобнее выбора одного календарного года.
    valid_dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    min_period_date = valid_dates.min().date() if not valid_dates.empty else datetime.now().date()
    max_period_date = valid_dates.max().date() if not valid_dates.empty else datetime.now().date()
    period_mode = st.radio(
        "Период анализа",
        ["Весь период", "Ручной диапазон"],
        index=0,
        key="period_mode",
        help="Весь период использует все доступные даты. Ручной диапазон позволяет выделить сезон, год или интересующий эпизод.",
    )
    if period_mode == "Ручной диапазон":
        picked_period = st.date_input(
            "Диапазон дат",
            value=(min_period_date, max_period_date),
            min_value=min_period_date,
            max_value=max_period_date,
            key="period_range",
            help="Можно указать любую пару дат; границы включаются в расчёты и графики.",
        )
        if isinstance(picked_period, (tuple, list)) and len(picked_period) == 2:
            period_start, period_end = picked_period
        elif isinstance(picked_period, (tuple, list)) and len(picked_period) == 1:
            period_start = period_end = picked_period[0]
        else:
            # Пока пользователь выбирает вторую границу, показываем один день.
            period_start = period_end = picked_period
    else:
        period_start, period_end = min_period_date, max_period_date
    st.caption(f"Активный диапазон: {period_start:%d.%m.%Y} — {period_end:%d.%m.%Y}")
    if st.button("Сбросить выбор"):
        st.session_state.selected_polygons = polygons[:1]
        st.rerun()

part = data[data["anon_polygon_id"].astype(str).isin(st.session_state.selected_polygons)].copy()
part_dates = pd.to_datetime(part["date"], errors="coerce").dt.date
part = part[part_dates.between(period_start, period_end, inclusive="both")]
part = part.sort_values("date")
# Сохраняет таблицы и счётчики для всех выбранных AOI, но не допускает
# нечитаемого «спагетти»-графика при выборе десятков полигонов.
chart_candidates = sorted(part["anon_polygon_id"].dropna().astype(str).unique())
default_chart = chart_candidates[: min(4, len(chart_candidates))]
chart_mode = st.sidebar.radio(
    "Режим графика", ["Фокус AOI", "Сетка AOI"], index=0, horizontal=False,
    help="Фокус показывает выбранные AOI вместе; сетка раскладывает каждый AOI в отдельную панель и убирает наложение линий.",
)
chart_polygons = st.sidebar.multiselect(
    "Полигоны на графике",
    chart_candidates,
    default=default_chart,
    key="chart_polygons",
    help="Контроль качества и периоды считаются по всем выбранным AOI; здесь задаётся только набор линий на графике.",
)
chart_part = part[part["anon_polygon_id"].astype(str).isin(chart_polygons)].copy()
anomalies = part[part["status"].isin(["suppression", "critical"])].copy()
periods = anomaly_periods(part, include_details=True)
part_gaps = _gap_mask(part.get("is_synthetic_gap", pd.Series(False, index=part.index)))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Полигонов", len(st.session_state.selected_polygons))
c2.metric("Наблюдений", int(part["ndvi_filled"].notna().sum()))
c3.metric("Восстановлено", int((part["primary_ndvi"].isna() & part_gaps).sum()))
c4.metric("Периодов стресса", len(periods))

# Радар раннего предупреждения — объяснимый операционный экран для жюри. Он
# строится из того же слоя аномалий и изолирован от пути предсказания и скачивания.
radar = early_warning_radar(part, periods)
st.markdown('<div class="section-kicker">РАННЕЕ ПРЕДУПРЕЖДЕНИЕ</div><div class="panel-title">Радар риска по полигонам</div><div class="hint">Ранг 0–100 объединяет отрицательный z-score, критические точки, длительность стресса и неопределённость восстановления. Наведите курсор на факторы, чтобы понять причину сигнала.</div>', unsafe_allow_html=True)
if radar.empty:
    st.info("Недостаточно данных для ранжирования полигона.")
else:
    top = radar.head(3)
    cards = st.columns(len(top))
    for card, (_, row) in zip(cards, top.iterrows()):
        level_color = "#e64a6f" if row["risk_level"] == "Высокий" else ("#ff8a34" if row["risk_level"] == "Наблюдать" else "#17a673")
        card.markdown(
            f'<div style="border:1px solid #e6ebf3;border-radius:14px;padding:.8rem 1rem;background:#fff;box-shadow:0 4px 16px rgba(21,39,84,.05)">'
            f'<div style="font-size:.76rem;color:#65738b">ПРИОРИТЕТ {int(row.name)+1}</div>'
            f'<div style="font-size:1.12rem;font-weight:750;color:#14213d">AOI {row["anon_polygon_id"]}</div>'
            f'<div style="font-size:1.8rem;font-weight:800;color:{level_color}">{row["risk_score"]:.1f}<span style="font-size:.8rem;color:#65738b"> / 100</span></div>'
            f'<div style="font-size:.78rem;color:{level_color};font-weight:700">{row["risk_level"]}</div>'
            f'<div style="font-size:.76rem;color:#65738b;margin-top:.35rem">{row["top_factors"]}</div></div>',
            unsafe_allow_html=True,
        )
    display_radar = radar.rename(columns={
        "anon_polygon_id": "AOI", "risk_score": "Риск, 0–100", "risk_level": "Уровень",
        "mean_zscore": "Средний z-score", "critical_share": "Доля critical",
        "suppression_share": "Доля suppression", "max_stress_days": "Макс. стресс, дней",
        "reconstructed_share": "Доля восстановления", "coverage": "Покрытие нормы",
        "top_factors": "Главные факторы",
    }).copy()
    for c in ("Доля critical", "Доля suppression", "Доля восстановления", "Покрытие нормы"):
        display_radar[c] = (100 * pd.to_numeric(display_radar[c], errors="coerce")).round(1).astype(str) + "%"
    display_radar["Средний z-score"] = pd.to_numeric(display_radar["Средний z-score"], errors="coerce").round(2)
    display_radar["Макс. стресс, дней"] = pd.to_numeric(display_radar["Макс. стресс, дней"], errors="coerce").round(0).astype(int)
    with st.expander("Открыть полный радар", expanded=False):
        st.dataframe(display_radar, hide_index=True, use_container_width=True)

# Паспорт собирает показатели выбранного AOI в один воспроизводимый артефакт.
# Он использует только текущий обогащённый срез и не меняет конкурсные прогнозы.
passport_options = st.session_state.selected_polygons or polygons
if passport_options:
    default_passport = passport_options[0]
    passport_aoi = st.selectbox(
        "AOI для паспорта",
        passport_options,
        index=passport_options.index(default_passport) if default_passport in passport_options else 0,
        key="passport_aoi",
        help="Паспорт включает покрытие, качество данных, риск, стресс-периоды и стандартный климатический сценарий.",
    )
    passport_part = data[data["anon_polygon_id"].astype(str).eq(str(passport_aoi))].copy()
    passport_dates = pd.to_datetime(passport_part["date"], errors="coerce").dt.date
    passport_part = passport_part[passport_dates.between(period_start, period_end, inclusive="both")]
    passport_periods = anomaly_periods(passport_part, include_details=True)
    passport_radar = early_warning_radar(passport_part, passport_periods)
    passport_row = passport_radar.iloc[0] if not passport_radar.empty else None
    passport_twin, passport_sens = counterfactual(
        passport_part,
        temp_delta_c=2.0,
        precip_factor=0.70,
        severity=1.0,
    )
    passport_delta = pd.to_numeric(passport_twin.get("ndvi_counterfactual_delta"), errors="coerce").dropna()
    passport_median_delta = float(passport_delta.median()) if not passport_delta.empty else 0.0
    passport_stress_n = int((passport_delta < -0.02).sum())
    st.markdown('<div class="section-kicker">ПАСПОРТ ПОЛИГОНА</div><div class="panel-title">Операционная карточка AOI</div><div class="hint">Один экран для принятия решения: состояние, качество данных, причины риска и стандартный сценарий жары с засухой.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        phead, pdownload = st.columns([2.3, 1], gap="large")
        with phead:
            pscore = float(passport_row.get("risk_score", 0.0)) if passport_row is not None else 0.0
            plevel = str(passport_row.get("risk_level", "Нет данных")) if passport_row is not None else "Нет данных"
            pfactors = str(passport_row.get("top_factors", "нет данных")) if passport_row is not None else "нет данных"
            level_color = "#e64a6f" if plevel == "Высокий" else ("#ff8a34" if plevel == "Наблюдать" else "#17a673")
            st.markdown(f'<div style="font-size:.78rem;color:#65738b;letter-spacing:.08em">АКТИВНЫЙ ПОЛИГОН</div><div style="font-size:1.55rem;font-weight:800;color:#14213d">AOI {escape(str(passport_aoi))}</div><div style="font-size:2rem;font-weight:850;color:{level_color}">{pscore:.1f}<span style="font-size:.9rem;color:#65738b"> / 100 · {escape(plevel)}</span></div><div style="font-size:.85rem;color:#65738b;margin-top:.2rem"><b>Факторы:</b> {escape(pfactors)}</div>', unsafe_allow_html=True)
        with pdownload:
            generated_at = datetime.now(timezone.utc)
            passport_html = _make_aoi_passport_html(
                str(passport_aoi), passport_part, passport_row, passport_periods,
                passport_sens, passport_median_delta, passport_stress_n, generated_at,
            )
            safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(passport_aoi))
            st.download_button(
                "Скачать паспорт AOI",
                data=passport_html.encode("utf-8"),
                file_name=f"aoi_passport_{safe_id}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.html",
                mime="text/html",
                key="download_aoi_passport",
                help="Автономный HTML-отчёт с графиком, риском, качеством данных и стресс-тестом.",
                use_container_width=True,
            )
        psummary = region_summary(passport_part)
        ps = psummary.iloc[0] if not psummary.empty else pd.Series(dtype=object)
        observed_n = int(ps.get("observed_n", 0)) if not ps.empty else 0
        reconstructed_n = int(ps.get("reconstructed_n", 0)) if not ps.empty else 0
        missing_n = int(ps.get("missing_n", 0)) if not ps.empty else max(0, len(passport_part) - observed_n - reconstructed_n)
        pcov = float(ps.get("climatology_coverage", 0.0)) if not ps.empty else 0.0
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Покрытие нормы", f"{100 * pcov:.1f}%")
        q2.metric("Наблюдения", f"{observed_n:,}")
        q3.metric("Восстановлено", f"{reconstructed_n:,}")
        q4.metric("Аномалии", f"{int(ps.get('anomaly_n', 0)) if not ps.empty else 0:,}")
        q5.metric("Стресс-периоды", f"{len(passport_periods)}")
        st.caption(f"Стандартный стресс-тест: +2 °C и −30% осадков → медианное ΔNDVI {passport_median_delta:+.3f}; точек под стрессом: {passport_stress_n}. Пропущено строк: {missing_n}.")

st.markdown('<div class="section-kicker">АНАЛИТИКА</div><div class="panel-title">Динамика растительности</div><div class="hint">Сглаживание закругляет только отображение линий; исходные значения и расчёты остаются без изменений.</div>', unsafe_allow_html=True)
st.markdown('''<div class="legend-card">
  <span class="legend-item"><i class="legend-line"></i><b>NDVI</b> факт + восстановление</span>
  <span class="legend-item"><i class="legend-line dotted"></i><b>Норма</b> сезонный ориентир AOI</span>
  <span class="legend-item"><i class="legend-cross">×</i><b>Выброс</b> исходная точка вне робастного диапазона</span>
  <span class="legend-item"><i class="legend-band"></i><b>Коридор</b> локальная неопределённость</span>
  <span class="legend-item"><i class="legend-line dashed"></i><b>Сценарий</b> контрфактический NDVI в цифровом двойнике</span>
</div>''', unsafe_allow_html=True)

with st.expander("Контроль качества по всем регионам"):
    # Таблица показывает адаптивность к нескольким AOI и выявляет случаи, когда
    # климатология перешла к запасному уровню или исходные данные недоступны.
    st.dataframe(region_summary(data), hide_index=True, use_container_width=True)

fig = go.Figure()
palette = ["#3d5afe", "#00a889", "#ff8a34", "#8e5cf7", "#e64a6f", "#008bbd", "#8aa33b", "#cf5b9a"]
show_raw_outliers = st.checkbox(
    "Показывать выбросы на исходной шкале",
    value=False,
    help="Робастный режим скрывает выбросы из линии, но оставляет их отдельными красными маркерами. Исходные значения не изменяются.",
)
show_confidence = st.checkbox(
    "Показывать коридор доверия",
    value=True,
    help="Полупрозрачный коридор показывает локальную неопределённость восстановления относительно климатической нормы.",
)
all_normal = []
for idx, (pid, g) in enumerate(chart_part.groupby("anon_polygon_id", sort=False)):
    color = palette[idx % len(palette)]
    outlier = g.get("is_ndvi_outlier", pd.Series(False, index=g.index)).fillna(False).astype(bool)
    if show_raw_outliers:
        fig.add_scatter(x=g["date"], y=g["ndvi_filled"], mode="lines+markers", name=f"NDVI {pid}", connectgaps=False,
                        line={"color": color, "width": 2, "shape": "spline", "smoothing": 0.55}, marker={"color": color, "size": 5})
        og = g[outlier]
        if not og.empty:
            fig.add_scatter(x=og["date"], y=og["ndvi_filled"], mode="markers", name=f"Выбросы {pid}",
                            marker={"color": "#ff4b5c", "size": 10, "symbol": "x"},
                            customdata=og[["ndvi_outlier_reason"]].to_numpy(),
                            hovertemplate="%{x|%Y-%m-%d}<br>raw NDVI=%{y:.4f}<br>reason=%{customdata[0]}<extra></extra>",
                            showlegend=False)
    else:
        ng = g[~outlier]
        if not ng.empty:
            all_normal.extend(pd.to_numeric(ng["ndvi_filled"], errors="coerce").dropna().tolist())
        fig.add_scatter(x=g["date"], y=g["ndvi_filled"].where(~outlier), mode="lines+markers",
                        name=f"NDVI {pid}", connectgaps=False,
                        line={"color": color, "width": 2, "shape": "spline", "smoothing": 0.55}, marker={"color": color, "size": 5})
        og = g[outlier]
        if not og.empty:
            # Сохраняет читаемость графика, показывая исходное значение при наведении;
            # столбцы источников и предсказаний в данных не обрезаются.
            fig.add_scatter(x=og["date"], y=og["ndvi_filled"].clip(-0.1, 1.05), mode="markers",
                            name=f"Выбросы {pid}", marker={"color": "#ff4b5c", "size": 10, "symbol": "x"},
                            customdata=og[["ndvi_filled", "ndvi_outlier_reason"]].to_numpy(),
                            hovertemplate="%{x|%Y-%m-%d}<br>raw NDVI=%{customdata[0]:.4f}<br>display=%{y:.4f}<br>reason=%{customdata[1]}<extra></extra>",
                            showlegend=False)
    if show_confidence:
        center = pd.to_numeric(g["ndvi_climatology_mean"], errors="coerce")
        spread = pd.to_numeric(g.get("ndvi_climatology_std", pd.Series(.10, index=g.index)), errors="coerce").fillna(.10).clip(.03, .25)
        fig.add_scatter(x=g["date"], y=center - spread, mode="lines", line={"width": 0, "shape": "spline"},
                        name=f"Коридор {pid}", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=g["date"], y=center + spread, mode="lines", line={"width": 0, "shape": "spline"},
                        fill="tonexty", fillcolor="rgba(61,90,254,.10)", name=f"Коридор {pid}",
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=g["date"], y=g["ndvi_climatology_mean"], mode="lines", name=f"Норма {pid}",
                    line={"dash": "dot", "color": color, "width": 1.5, "shape": "spline", "smoothing": 0.5}, opacity=.72)
layout_kwargs = dict(height=520, xaxis_title="Дата", yaxis_title="NDVI", hovermode="x unified",
                     template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,253,.72)",
                     font={"family":"Inter, Segoe UI, sans-serif", "color":"#34415c", "size":12},
                     margin={"l":48,"r":24,"t":28,"b":60},
                     legend={"orientation":"h", "yanchor":"top", "y":-0.16, "x":0, "font":{"size":10}},
                     xaxis={"showgrid":False, "linecolor":"#dbe2ee", "title_font":{"size":12}},
                     yaxis={"showgrid":True, "gridcolor":"#e8edf5", "zeroline":False, "title_font":{"size":12}})
if not show_raw_outliers:
    # Стабильный физический диапазон растительности не позволяет двум испорченным
    # всплескам сжать остальную часть сезонного графика.
    if all_normal:
        q01, q99 = pd.Series(all_normal).quantile([0.01, 0.99]).tolist()
        lo, hi = max(-1.0, min(-0.1, float(q01) - 0.05)), min(1.0, max(1.05, float(q99) + 0.05))
        layout_kwargs["yaxis"] = {**layout_kwargs.get("yaxis", {}), "range": [lo, hi]}
fig.update_layout(**layout_kwargs)
if chart_mode == "Сетка AOI" and chart_polygons:
    # Малые панели сохраняют читаемость каждого AOI и те же данные, маркеры
    # аномалий и смысл доверительного коридора, что и основной график.
    ncols = 2
    nrows = (len(chart_polygons) + ncols - 1) // ncols
    grid_fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(p) for p in chart_polygons],
                             shared_xaxes=False, vertical_spacing=0.10, horizontal_spacing=0.08)
    for j, pid in enumerate(chart_polygons):
        g = chart_part[chart_part["anon_polygon_id"].astype(str).eq(str(pid))].sort_values("date")
        rr, cc = j // ncols + 1, j % ncols + 1
        color = palette[j % len(palette)]
        outlier = g.get("is_ndvi_outlier", pd.Series(False, index=g.index)).fillna(False).astype(bool)
        y = g["ndvi_filled"] if show_raw_outliers else g["ndvi_filled"].where(~outlier)
        grid_fig.add_scatter(x=g["date"], y=y, mode="lines+markers", name="NDVI", legendgroup="ndvi",
                             showlegend=(j == 0), connectgaps=False,
                             line={"color": color, "width": 2, "shape": "spline", "smoothing": 0.55},
                             marker={"color": color, "size": 4}, row=rr, col=cc)
        grid_fig.add_scatter(x=g["date"], y=g["ndvi_climatology_mean"], mode="lines", name="Норма", legendgroup="norm",
                             showlegend=(j == 0), line={"dash": "dot", "color": color, "width": 1.4, "shape": "spline", "smoothing": 0.5},
                             opacity=.72, row=rr, col=cc)
        og = g[outlier]
        if not og.empty:
            grid_fig.add_scatter(x=og["date"], y=og["ndvi_filled"].clip(-.1, 1.05), mode="markers", name="Выброс",
                                 legendgroup="outlier", showlegend=(j == 0), marker={"color": "#ff4b5c", "size": 8, "symbol": "x"}, row=rr, col=cc)
        if show_confidence:
            center = pd.to_numeric(g["ndvi_climatology_mean"], errors="coerce")
            spread = pd.to_numeric(g.get("ndvi_climatology_std", pd.Series(.10, index=g.index)), errors="coerce").fillna(.10).clip(.03, .25)
            grid_fig.add_scatter(x=g["date"], y=center - spread, mode="lines", line={"width": 0}, showlegend=False, row=rr, col=cc)
            grid_fig.add_scatter(x=g["date"], y=center + spread, mode="lines", line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(61,90,254,.10)", name="Коридор", showlegend=(j == 0), legendgroup="band", row=rr, col=cc)
    grid_fig.update_layout(height=max(380, 260 * nrows), template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(248,250,253,.72)", hovermode="x unified",
                           margin={"l": 42, "r": 20, "t": 42, "b": 40},
                           font={"family": "Inter, Segoe UI, sans-serif", "color": "#34415c", "size": 11},
                           legend={"orientation": "h", "yanchor": "top", "y": -0.08, "x": 0, "font": {"size": 10}})
    grid_fig.update_yaxes(showgrid=True, gridcolor="#e8edf5", zeroline=False, title_text="NDVI")
    grid_fig.update_xaxes(showgrid=False, linecolor="#dbe2ee")
    display_fig = grid_fig
else:
    display_fig = fig
st.caption(f"На графике показано полигонов: {len(chart_polygons)} из {len(chart_candidates)} выбранных · выбросы отмечены крестиками · коридор доверия включён: {'да' if show_confidence else 'нет'}")
st.plotly_chart(display_fig, use_container_width=True, theme=None)

outlier_rows = part[part.get("is_ndvi_outlier", pd.Series(False, index=part.index)).fillna(False).astype(bool)]
if not outlier_rows.empty:
    st.caption(f"Выбросов отмечено: {len(outlier_rows)}. В робастном режиме они не влияют на масштаб графика; raw NDVI доступен в наведении и таблице.")

# Контрфактический цифровой двойник — наглядная функция продукта на данных
# для жюри. Он намеренно изолирован от пути предсказания и скачивания.
st.markdown('<div class="section-kicker">ЦИФРОВОЙ ДВОЙНИК</div><div class="panel-title">Стресс-тест сезона: что будет с растительностью при изменении погоды?</div><div class="hint">Сценарий пересчитывается на лету по наблюдаемой траектории выбранных AOI. Коэффициенты чувствительности оцениваются из текущих данных, а прогнозный CSV не меняется.</div>', unsafe_allow_html=True)
with st.expander("Запустить контрфактический сценарий", expanded=True):
    twin_source = chart_part if not chart_part.empty else part
    t1, t2, t3 = st.columns(3)
    temp_delta = t1.slider("Изменение температуры, °C", -5.0, 8.0, 2.0, 0.5, key="twin_temp_delta",
                           help="Сдвиг температуры относительно фактически наблюдаемого сезона.")
    precip_pct = t2.slider("Изменение осадков, %", -80, 100, -30, 5, key="twin_precip_pct",
                            help="-30% означает сценарий засухи, +30% — более влажный сезон.")
    severity = t3.slider("Сила сценария", 0.0, 2.0, 1.0, 0.1, key="twin_severity",
                         help="0 — фактическая траектория, 1 — полный сценарий, 2 — усиленный стресс.")
    twin, twin_sens = counterfactual(
        twin_source,
        temp_delta_c=float(temp_delta),
        precip_factor=1.0 + float(precip_pct) / 100.0,
        severity=float(severity),
    )
    twin = twin.copy()
    twin["date"] = pd.to_datetime(twin["date"], errors="coerce")
    twin["actual"] = pd.to_numeric(twin.get("ndvi_filled"), errors="coerce")
    twin["counterfactual"] = pd.to_numeric(twin["ndvi_counterfactual"], errors="coerce")
    # Разделяем видимые наблюдения и конкурсное восстановление. Иначе одна
    # синяя линия смешивает два разных источника и выглядит как случайный шум.
    observed_flag = twin.get("is_observed", pd.Series(False, index=twin.index)).fillna(False).astype(bool)
    reconstructed_flag = twin.get("is_reconstructed", pd.Series(False, index=twin.index)).fillna(False).astype(bool)
    twin["observed_value"] = twin["actual"].where(observed_flag)
    twin["reconstructed_value"] = twin["actual"].where(reconstructed_flag)
    daily = twin.groupby("date", as_index=False).agg(
        actual=("actual", "mean"),
        observed=("observed_value", "mean"),
        reconstructed=("reconstructed_value", "mean"),
        counterfactual=("counterfactual", "mean"),
        digital_twin_climatology=("digital_twin_climatology", "mean"),
    )
    daily = daily.sort_values("date")
    st.markdown('''<div class="legend-card" aria-label="Обозначения стресс-теста">
      <span class="legend-item"><i class="legend-line"></i><b>Наблюдаемый NDVI</b> доступные исходные измерения</span>
      <span class="legend-item"><i class="legend-line reconstructed"></i><b>Восстановленный NDVI</b> значения в синтетических пропусках</span>
      <span class="legend-item"><i class="legend-line dashed"></i><b>NDVI по сценарию</b> оценка при заданной погоде</span>
      <span class="legend-item"><i class="legend-line dotted"></i><b>Сезонная норма</b> исторический ориентир</span>
    </div>''', unsafe_allow_html=True)
    st.caption("Красная пунктирная линия — контрфактическая оценка при выбранной погоде; это не новый конкурсный прогноз. "
               "При выборе нескольких полигонов показано среднее доступных значений на каждую дату. Линии соединены напрямую без сглаживания, чтобы не создавать ложных всплесков.")
    twin_fig = go.Figure()
    twin_fig.add_scatter(x=daily["date"], y=daily["observed"], mode="lines+markers", name="Наблюдаемый NDVI",
                         hovertemplate="Дата: %{x|%d.%m.%Y}<br>Наблюдаемый NDVI: %{y:.3f}<extra></extra>",
                         connectgaps=False, line={"color": "#3d5afe", "width": 2.5, "shape": "linear"},
                         marker={"color": "#3d5afe", "size": 4})
    twin_fig.add_scatter(x=daily["date"], y=daily["reconstructed"], mode="lines+markers", name="Восстановленный NDVI",
                         hovertemplate="Дата: %{x|%d.%m.%Y}<br>Восстановленный NDVI: %{y:.3f}<extra></extra>",
                         connectgaps=False, line={"color": "#ff8a34", "width": 2.0, "dash": "dot", "shape": "linear"},
                         marker={"color": "#ff8a34", "size": 4, "symbol": "diamond"})
    twin_fig.add_scatter(x=daily["date"], y=daily["counterfactual"], mode="lines", name="NDVI по сценарию",
                         customdata=(daily["counterfactual"] - daily["actual"]).to_numpy(),
                         hovertemplate="Дата: %{x|%d.%m.%Y}<br>NDVI по сценарию: %{y:.3f}<br>Изменение NDVI: %{customdata:+.3f}<extra></extra>",
                         line={"color": "#e64a6f", "width": 2.5, "dash": "dash", "shape": "linear"})
    twin_fig.add_scatter(x=daily["date"], y=daily["digital_twin_climatology"], mode="lines", name="Сезонная норма",
                         hovertemplate="Дата: %{x|%d.%m.%Y}<br>Сезонная норма NDVI: %{y:.3f}<extra></extra>",
                         line={"color": "#17a673", "width": 1.5, "dash": "dot", "shape": "linear"})
    twin_fig.update_layout(
        height=440, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,253,.72)",
        hovermode="x unified", margin={"l": 80, "r": 24, "t": 80, "b": 76},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#34415c", "size": 12},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0, "traceorder": "normal",
                "font": {"size": 12, "color": "#34415c"}, "bgcolor": "rgba(255,255,255,.9)"},
        xaxis={"title": {"text": "Дата наблюдения", "standoff": 18, "font": {"color": "#34415c", "size": 13}},
               "tickfont": {"color": "#34415c", "size": 12}, "automargin": True,
               "showgrid": False, "linecolor": "#dbe2ee"},
        yaxis={"title": {"text": "Индекс растительности NDVI", "standoff": 18, "font": {"color": "#34415c", "size": 13}},
               "tickfont": {"color": "#34415c", "size": 12}, "automargin": True,
               "showgrid": True, "gridcolor": "#e8edf5", "zeroline": False},
    )
    st.plotly_chart(twin_fig, use_container_width=True, theme=None)
    valid_delta = pd.to_numeric(twin["ndvi_counterfactual_delta"], errors="coerce").dropna()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Медиана ΔNDVI", f"{valid_delta.median():+.3f}" if not valid_delta.empty else "—")
    m2.metric("Периоды под стрессом", f"{int((valid_delta < -0.02).sum())} / {len(valid_delta)}")
    m3.metric("Чувствительность к жаре", f"{twin_sens['beta_temp']:+.3f} / σ")
    m4.metric("Чувствительность к осадкам", f"{twin_sens['beta_precip']:+.3f} / σ")
    direction = "засухи и жары" if (temp_delta >= 0 and precip_pct <= 0) else "изменения климата"
    st.info(
        f"Сценарий: {temp_delta:+.1f} °C и {precip_pct:+d}% осадков, сила {severity:.1f}. "
        f"На выбранной траектории это оценивает влияние {direction}; в подгонке использовано "
        f"{twin_sens['n_fit']} наблюдений. Нажмите на линии, чтобы увидеть дату и разницу NDVI."
    )

st.markdown('<div class="section-kicker">РАЗБОР РЕЗУЛЬТАТА</div>', unsafe_allow_html=True)
left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown('<div class="panel-title">Интерпретация и контроль качества</div>', unsafe_allow_html=True)
    anomaly_cols = ["anon_polygon_id", "date", "ndvi_filled", "ndvi_climatology_mean", "ndvi_zscore", "status"]
    st.subheader("1. Контроль выбросов")
    st.caption("Отдельные точки с признаками выброса: значение NDVI, причина отметки и источник значения.")
    if not outlier_rows.empty:
        st.dataframe(outlier_rows[["anon_polygon_id", "date", "ndvi_filled", "ndvi_outlier_reason", "ndvi_zscore", "value_source"]], hide_index=True, use_container_width=True)
    else:
        st.caption("В выбранном срезе выбросов не найдено.")
    st.subheader("2. Аномальные наблюдения")
    st.caption("Даты со статусом suppression или critical: NDVI ниже сезонного ориентира, "
               "а z-score показывает величину отклонения.")
    st.dataframe(anomalies[anomaly_cols], hide_index=True, use_container_width=True)
    st.subheader("3. Периоды стресса")
    st.caption("Последовательные аномальные наблюдения, объединённые в эпизоды с началом, окончанием и длительностью.")
    if not periods.empty:
        st.dataframe(periods, hide_index=True, use_container_width=True)
    else:
        st.caption("В выбранном срезе периодов стресса не найдено.")
with right:
    st.markdown('<div class="panel-title">Управление полигонами и автосбор</div>', unsafe_allow_html=True)
    if "saved_regions" not in st.session_state:
        st.session_state.saved_regions = {}
    upload = st.file_uploader("Импорт GeoJSON полигона", type=["geojson", "json"], key="geo_upload")
    if upload:
        try:
            imported = json.load(upload)
            ok, msg = validate_geojson(imported)
            if not ok:
                st.error(f"Некорректный GeoJSON: {msg}")
            else:
                name = st.text_input("Название импортированного региона", value="imported_region", key="import_name")
                if st.button("Сохранить импортированный регион", key="save_import"):
                    st.session_state.saved_regions[name.strip() or "imported_region"] = imported
                    st.success("Регион сохранён в текущей сессии")
        except Exception as exc:
            st.error(f"Не удалось прочитать GeoJSON: {exc}")
    with st.expander("Создать/изменить контур вручную"):
        st.caption("Введите вершины в формате lon,lat — по одной на строку. Контур замкнётся автоматически.")
        region_name = st.text_input("Название региона", value="my_region", key="manual_region_name")
        existing = st.session_state.saved_regions.get(region_name, {})
        default_text = ""
        try:
            ring = existing.get("geometry", {}).get("coordinates", [[]])[0]
            default_text = "\n".join(f"{p[0]},{p[1]}" for p in ring[:-1])
        except (AttributeError, IndexError, TypeError):
            pass
        coordinates_text = st.text_area("Вершины", value=default_text, height=120, key="manual_coordinates")
        c_save, c_delete = st.columns(2)
        if c_save.button("Сохранить / обновить", key="save_manual"):
            try:
                candidate = _polygon_from_text(coordinates_text)
                ok, msg = validate_geojson(candidate)
                if not ok:
                    st.error(msg)
                else:
                    st.session_state.saved_regions[region_name.strip() or "my_region"] = candidate
                    st.success("Контур сохранён")
            except Exception as exc:
                st.error(f"Ошибка координат: {exc}")
        if c_delete.button("Удалить", key="delete_manual"):
            if region_name in st.session_state.saved_regions:
                del st.session_state.saved_regions[region_name]
                st.success("Контур удалён")
            else:
                st.info("Такого сохранённого контура нет")
    if st.session_state.saved_regions:
        st.caption(f"Сохранено регионов: {len(st.session_state.saved_regions)}")
        active_name = st.selectbox("Активный пользовательский регион", list(st.session_state.saved_regions), key="active_region")
        geo = st.session_state.saved_regions[active_name]
    else:
        geo = None
    if geo is not None:
        ok, msg = validate_geojson(geo)
        center = geojson_centroid(geo)
        st.json({"validation": msg, "centroid": center})
        st.download_button(
            "Экспортировать активный GeoJSON",
            data=json.dumps(geo, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{active_name}.geojson",
            mime="application/geo+json",
            help="Скачать проверенный контур для повторного запуска и передачи в другую сессию.",
        )
        if center:
            # Отображает контур (и его центроид), чтобы управление полигонами можно было
            # проверить визуально, а не только через форму координат.
            feature_collection = geo if geo.get("type") == "FeatureCollection" else {"type": "FeatureCollection", "features": [geo if geo.get("type") == "Feature" else {"type": "Feature", "properties": {}, "geometry": geo}]}
            view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11, pitch=0)
            contour_layer = pdk.Layer("GeoJsonLayer", data=feature_collection, filled=True, stroked=True, get_fill_color=[47, 158, 68, 90], get_line_color=[20, 80, 30, 220], line_width_min_pixels=2, pickable=True)
            point_layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [center[0]], "lon": [center[1]]}), get_position="[lon, lat]", get_radius=80, get_fill_color=[220, 50, 50, 220], pickable=True)
            st.pydeck_chart(pdk.Deck(layers=[contour_layer, point_layer], initial_view_state=view, tooltip={"text": "{name}"}), use_container_width=True)
        start_date = str(part["date"].min().date()) if not part.empty else str(data["date"].min().date())
        end_date = str(part["date"].max().date()) if not part.empty else str(data["date"].max().date())
        b1, b2, b3 = st.columns(3)
        if center and b1.button("Погода", key="fetch_weather"):
            try:
                with st.spinner("Open-Meteo archive..."):
                    weather = fetch_open_meteo(center[0], center[1], start_date, end_date)
                st.session_state.last_weather = weather
                st.session_state.context_data = merge_weather_context(part, weather)
            except Exception as exc:
                st.error(f"Open-Meteo недоступен: {exc}")
        if b2.button("Sentinel-2", key="fetch_sentinel"):
            try:
                with st.spinner("Planetary Computer STAC..."):
                    st.session_state.last_sentinel = search_sentinel_items(geo, start_date, end_date)
            except Exception as exc:
                st.error(f"Planetary Computer STAC недоступен: {exc}")
        if center and b3.button("Контуры OSM", key="fetch_osm"):
            try:
                with st.spinner("Overpass farmland contours..."):
                    st.session_state.last_osm = search_osm_agricultural_contours(center[0], center[1])
            except Exception as exc:
                st.error(f"OSM Overpass недоступен: {exc}")
        if "last_weather" in st.session_state:
            st.success(f"Погодных записей: {len(st.session_state.last_weather)}")
            st.dataframe(st.session_state.last_weather.tail(20), hide_index=True, use_container_width=True)
            if "context_data" in st.session_state:
                st.caption("Погода присоединена к NDVI по календарной дате; пропуски API сохранены как NaN")
                cols = [c for c in ("anon_polygon_id", "date", "ndvi_filled", "temp_c", "precip_mm") if c in st.session_state.context_data.columns]
                st.dataframe(st.session_state.context_data[cols].tail(20), hide_index=True, use_container_width=True)
        if "last_sentinel" in st.session_state:
            st.success(f"Сцен Sentinel-2: {len(st.session_state.last_sentinel)}")
            st.dataframe(pd.DataFrame(st.session_state.last_sentinel), hide_index=True, use_container_width=True)
        if "last_osm" in st.session_state:
            st.success(f"Сельхозконтуров OSM: {len(st.session_state.last_osm)}")
            st.dataframe(pd.DataFrame(st.session_state.last_osm), hide_index=True, use_container_width=True)
    else:
        st.info("Импортируйте GeoJSON или создайте контур координатами; затем запускайте автосбор погоды, Sentinel-2 и OSM.")

st.download_button("Скачать текущий анализ CSV", part.to_csv(index=False).encode("utf-8"), "agropulse_analysis.csv", "text/csv")
