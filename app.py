# app.py — Relative Strength Scanner (standalone)
# Universe: Tickers.csv (your ~6,695 stocks)
# Benchmark: SPY
#
# Notes:
# - Designed to scan thousands of tickers reliably by downloading prices in batches.
# - RS ratings (1–99) are percentile ranks vs YOUR scanned universe (Tickers.csv).
# - No RS dashboard sections. No Big Picture Market Pulse. Scanner only.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Relative Strength Scanner", layout="wide")

BENCHMARK_DEFAULT = "SPY"
DEFAULT_UNIVERSE_CSV = "Tickers.csv"

SPARK_CHARS = "▁▂▃▄▅▆▇█"

# =========================
# HELPERS
# =========================
def _asof_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _normalize_ticker(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().upper()
    if not t:
        return ""
    # yfinance uses '-' for class shares: BRK-B, BF-B, etc.
    t = t.replace(".", "-")
    return t


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _chunked(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# =========================
# CSS (matches your dashboard vibe)
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.card h3{margin:0 0 8px 0; font-size: 1.02rem; font-weight: 950;}
.card .hint{opacity:0.72; font-size:0.88rem; margin-top:-2px; margin-bottom:10px;}

.badge{
  display:inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.78rem;
  letter-spacing: 0.2px;
  border: 1px solid rgba(255,255,255,0.12);
}
.badge-yes{background: rgba(124,252,154,0.15); color:#7CFC9A;}
.badge-no{background: rgba(255,107,107,0.12); color:#FF6B6B;}
.badge-neutral{background: rgba(255,200,60,0.12); color: rgba(255,200,60,0.98);}

.pill{
  display:inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 950;
  font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.12);
}
.pill-red{background: rgba(255,80,80,0.16); color:#FF6B6B;}
.pill-amber{background: rgba(255,200,60,0.16); color: rgba(255,200,60,0.98);}
.pill-green{background: rgba(80,255,120,0.16); color:#7CFC9A;}

.pl-table-wrap {border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10);}
table.pl-table {border-collapse: collapse; width: 100%; font-size: 13px;}
table.pl-table thead th {
  position: sticky; top: 0;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  font-weight: 900;
}
table.pl-table tbody td{
  padding: 7px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: middle;
}
td.mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
td.ticker {font-weight: 900;}
td.name {white-space: normal; line-height: 1.15;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner=False)
def load_universe_from_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))

    # Find a ticker column
    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("symbols") or cols.get("tickers")

    if not ticker_col:
        # fallback: if first col looks like tickers
        ticker_col = df.columns[0]

    out = df.copy()
    out["Ticker"] = out[ticker_col].astype(str).map(_normalize_ticker)
    out = out[(out["Ticker"] != "") & (out["Ticker"].notna())]
    out = out.drop_duplicates(subset=["Ticker"])
    return out


@st.cache_data(show_spinner=False)
def load_universe_from_repo_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("symbols") or cols.get("tickers")

    if not ticker_col:
        ticker_col = df.columns[0]

    out = df.copy()
    out["Ticker"] = out[ticker_col].astype(str).map(_normalize_ticker)
    out = out[(out["Ticker"] != "") & (out["Ticker"].notna())]
    out = out.drop_duplicates(subset=["Ticker"])
    return out


# -----------------------------
# Batched price pulls (reliable for thousands)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_prices_batched(
    tickers: List[str],
    period: str,
    batch_size: int,
) -> pd.DataFrame:
    """
    Returns a close-price dataframe (columns=tickers), ffilled, with all-NaN columns dropped.
    Uses yfinance in batches to avoid huge single requests failing.
    """
    tickers = [t for t in tickers if t]
    tickers = _dedupe_keep_order(tickers)

    frames = []
    batches = _chunked(tickers, batch_size)

    # NOTE: streamlit progress can't run inside cached function,
    # so the cached function is purely computation.
    # Progress is handled by a non-cached wrapper below.
    for b in batches:
        df = yf.download(
            tickers=b,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            closes = {}
            for t in b:
                if (t, "Close") in df.columns:
                    closes[t] = df[(t, "Close")]
            close_df = pd.DataFrame(closes)
        else:
            # Single ticker case
            t0 = b[0]
            if "Close" in df.columns:
                close_df = pd.DataFrame({t0: df["Close"]})
            else:
                continue

        frames.append(close_df)

    if not frames:
        raise RuntimeError("No data returned from price source for any batch.")

    close_all = pd.concat(frames, axis=1)
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]
    close_all = close_all.dropna(how="all").ffill()
    return close_all


def fetch_prices_batched_with_progress(
    tickers: List[str],
    period: str,
    batch_size: int,
) -> pd.DataFrame:
    """
    Wrapper around the cached function that also shows progress while building
    the cache on first run. If cache hits, this returns fast.
    """
    # If cached already, this returns immediately and we won't show progress.
    # But we can still show a short spinner for UX.
    with st.spinner("Pulling price data (cached when possible)..."):
        return fetch_prices_batched(tickers, period=period, batch_size=batch_size)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_names_for_results(tickers: List[str]) -> Dict[str, str]:
    """
    Only fetch names for the final result set (fast).
    """
    names = {t: t for t in tickers}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            n = info.get("shortName") or info.get("longName")
            if n:
                names[t] = str(n)
        except Exception:
            pass
    return names


# =========================
# RS CALCS
# =========================
HORIZONS_RS = {
    "RS 1W": 5,
    "RS 1M": 21,
    "RS 3M": 63,
    "RS 6M": 126,
    "RS 1Y": 252,
}


def _ratio_rs(close_t: pd.Series, close_b: pd.Series, periods: int) -> pd.Series:
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1


def _calc_rs_raw(close_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    b = close_df[benchmark]
    rows = []

    for t in close_df.columns:
        if t == benchmark:
            continue
        s = close_df[t]
        rec = {"Ticker": t, "__has_data": int(s.dropna().shape[0] > 0)}
        for col, n in HORIZONS_RS.items():
            rr = _ratio_rs(s, b, n)
            rec[col] = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def _to_percentile_1_99(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = (s.rank(pct=True) * 99).round().clip(1, 99)
    return out


def _accel_flag(row: pd.Series) -> bool:
    # Accelerating: short-term RS >= longer-term RS (improving into the present)
    vals = [row.get("RS 1W"), row.get("RS 1M"), row.get("RS 3M"), row.get("RS 6M"), row.get("RS 1Y")]
    if any(pd.isna(v) for v in vals):
        return False
    return vals[0] >= vals[1] >= vals[2] >= vals[3] >= vals[4]


def _decel_flag(row: pd.Series) -> bool:
    # Decelerating: short-term RS <= longer-term RS (losing momentum recently)
    vals = [row.get("RS 1W"), row.get("RS 1M"), row.get("RS 3M"), row.get("RS 6M"), row.get("RS 1Y")]
    if any(pd.isna(v) for v in vals):
        return False
    return vals[0] <= vals[1] <= vals[2] <= vals[3] <= vals[4]


# =========================
# SPARKLINE (optional visual, like your dashboard)
# =========================
def sparkline_from_series(s: pd.Series, n: int = 26) -> Tuple[str, List[int]]:
    s = s.dropna().tail(n)
    if s.empty:
        return "", []
    if s.nunique() == 1:
        mid = len(SPARK_CHARS) // 2
        return (SPARK_CHARS[mid] * len(s)), ([mid] * len(s))

    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return "", []
    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS) - 1)).round().astype(int).clip(0, len(SPARK_CHARS) - 1)
    levels = idx.tolist()
    spark = "".join(SPARK_CHARS[i] for i in levels)
    return spark, levels


def spark_html(spark: str, levels: List[int]) -> str:
    if not spark or not levels or len(spark) != len(levels):
        return ""

    def level_to_rgb(lv: int):
        t = lv / 7.0
        if t <= 0.5:
            k = t / 0.5
            r1, g1, b1 = 255, 80, 80
            r2, g2, b2 = 255, 200, 60
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        else:
            k = (t - 0.5) / 0.5
            r1, g1, b1 = 255, 200, 60
            r2, g2, b2 = 80, 255, 120
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        return r, g, b

    spans = []
    for ch, lv in zip(spark, levels):
        r, g, b = level_to_rgb(int(lv))
        spans.append(f'<span style="color: rgb({r},{g},{b}); font-weight:900;">{ch}</span>')
    return "".join(spans)


# =========================
# TABLE RENDERING
# =========================
def rs_bg(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    x = (v - 1) / 98.0
    if x < 0.5:
        r = 255
        g = int(80 + (x / 0.5) * (180 - 80))
    else:
        r = int(255 - ((x - 0.5) / 0.5) * (255 - 40))
        g = 200
    b = 60
    return (
        f"background-color: rgb({r},{g},{b}); color:#0B0B0B; "
        f"font-weight:900; border-radius:6px; padding:2px 6px; "
        f"display:inline-block; min-width:32px; text-align:center;"
    )


def fmt_price(v):
    try:
        return f"${float(v):,.2f}"
    except:
        return ""


def render_table_html(df: pd.DataFrame, columns: List[str], height_px: int = 760):
    th = "".join([f"<th>{c}</th>" for c in columns])

    trs = []
    for _, row in df.iterrows():
        tds = []
        for c in columns:
            val = row.get(c, "")
            td_class = ""
            if c == "Ticker":
                td_class = "ticker"
            elif c == "Name":
                td_class = "name"
            elif c in ("Price", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y", "RS Trend", "RS Spark"):
                td_class = "mono"

            if c == "Price":
                cell_html = fmt_price(val)
            elif c.startswith("RS "):
                if pd.isna(val):
                    cell_html = ""
                else:
                    txt = f"{float(val):.0f}"
                    stl = rs_bg(val)
                    cell_html = f'<span style="{stl}">{txt}</span>'
            elif c == "RS Trend":
                if val == "ACCEL":
                    cell_html = '<span class="badge badge-yes">ACCEL</span>'
                elif val == "DECEL":
                    cell_html = '<span class="badge badge-no">DECEL</span>'
                else:
                    cell_html = '<span class="badge badge-neutral">—</span>'
            elif c == "RS Spark":
                cell_html = spark_html(str(val), row.get("__spark_levels", []))
            else:
                cell_html = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)

            tds.append(f'<td class="{td_class}">{cell_html}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")

    table = f"""
    <div class="pl-table-wrap" style="max-height:{height_px}px; overflow:auto;">
      <table class="pl-table">
        <thead><tr>{th}</tr></thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)


# =========================
# UI
# =========================
st.title("Relative Strength Scanner")
st.caption(f"As of: {_asof_ts()} • Benchmark: {BENCHMARK_DEFAULT}")

with st.sidebar:
    st.subheader("Scanner Controls")

    # Universe source
    st.markdown("**Universe Source**")
    use_upload = st.toggle("Upload CSV instead of repo Tickers.csv", value=False)
    uploaded = None
    if use_upload:
        uploaded = st.file_uploader("Upload tickers CSV", type=["csv"])
        st.caption("CSV needs a column named Ticker or Symbol (or first column is used).")

    st.markdown("---")

    benchmark = st.text_input("Benchmark", value=BENCHMARK_DEFAULT).strip().upper() or BENCHMARK_DEFAULT

    price_period = st.selectbox("Price History Period", ["1y", "2y", "5y"], index=1)
    batch_size = st.selectbox("Download Batch Size", [100, 150, 200, 250, 300], index=2)
    st.caption("If scans fail/time out, lower batch size (e.g., 100–150).")

    st.markdown("---")
    st.markdown("**Filters**")

    filter_mode = st.selectbox(
        "Scan Mode",
        [
            "Any (no trend filter)",
            "Accelerating (RS 1W ≥ 1M ≥ 3M ≥ 6M ≥ 1Y)",
            "Decelerating (RS 1W ≤ 1M ≤ 3M ≤ 6M ≤ 1Y)",
        ],
        index=0,
    )

    min_rs_1w = st.slider("Min RS 1W", 1, 99, 90)
    min_rs_1m = st.slider("Min RS 1M", 1, 99, 90)
    min_rs_3m = st.slider("Min RS 3M", 1, 99, 80)
    min_rs_6m = st.slider("Min RS 6M", 1, 99, 70)
    min_rs_1y = st.slider("Min RS 1Y", 1, 99, 70)

    sort_col = st.selectbox("Sort By", ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"], index=1)
    top_n = st.slider("Max Results", 25, 500, 200, step=25)

    st.markdown("---")
    if st.button("Refresh (clear cache)", use_container_width=True):
        fetch_prices_batched.clear()
        fetch_names_for_results.clear()
        load_universe_from_repo_csv.clear()
        load_universe_from_csv_bytes.clear()
        st.rerun()

# =========================
# LOAD UNIVERSE
# =========================
try:
    if use_upload:
        if uploaded is None:
            st.warning("Upload a CSV to run the scan (or toggle off upload to use repo Tickers.csv).")
            st.stop()
        universe_df = load_universe_from_csv_bytes(uploaded.getvalue())
    else:
        universe_df = load_universe_from_repo_csv(DEFAULT_UNIVERSE_CSV)
except Exception as e:
    st.error(f"Failed to load universe: {e}")
    st.stop()

universe = universe_df["Ticker"].astype(str).map(_normalize_ticker).tolist()
universe = [t for t in universe if t]
universe = _dedupe_keep_order(universe)

# Ensure benchmark included
if benchmark not in universe:
    universe_plus = universe + [benchmark]
else:
    universe_plus = universe

# Summary card
st.markdown(
    f"""
<div class="card">
  <h3>Universe</h3>
  <div class="hint">RS Ratings are percentile-ranked (1–99) vs your selected universe.</div>
  <div class="small-muted">Tickers in universe: <b>{len(universe):,}</b> • Benchmark: <b>{benchmark}</b> • History: <b>{price_period}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# FETCH PRICES
# =========================
try:
    prices = fetch_prices_batched_with_progress(universe_plus, period=price_period, batch_size=int(batch_size))
except Exception as e:
    st.error(f"Data pull failed: {e}")
    st.stop()

if benchmark not in prices.columns:
    st.error(f"Benchmark '{benchmark}' did not return data from yfinance. Try SPY, QQQ, DIA, IWM, etc.")
    st.stop()

# Drop tickers that have no usable data
valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= 260]  # ~1y trading days
if benchmark not in valid_cols:
    valid_cols.append(benchmark)
prices = prices[valid_cols].copy()

# =========================
# CALC RS (raw then ranked)
# =========================
rs_raw = _calc_rs_raw(prices, benchmark=benchmark)
rs_ranked = _to_percentile_1_99(rs_raw, list(HORIZONS_RS.keys()))

# Add accel/decel tags
rs_ranked["__accel"] = rs_ranked.apply(_accel_flag, axis=1)
rs_ranked["__decel"] = rs_ranked.apply(_decel_flag, axis=1)

def _trend_label(r):
    if bool(r.get("__accel")):
        return "ACCEL"
    if bool(r.get("__decel")):
        return "DECEL"
    return ""

rs_ranked["RS Trend"] = rs_ranked.apply(_trend_label, axis=1)

# Apply minimum RS filters
f = rs_ranked.copy()
f = f[
    (f["RS 1W"] >= min_rs_1w)
    & (f["RS 1M"] >= min_rs_1m)
    & (f["RS 3M"] >= min_rs_3m)
    & (f["RS 6M"] >= min_rs_6m)
    & (f["RS 1Y"] >= min_rs_1y)
].copy()

# Apply mode filter
if filter_mode.startswith("Accelerating"):
    f = f[f["__accel"] == True].copy()
elif filter_mode.startswith("Decelerating"):
    f = f[f["__decel"] == True].copy()

# Sort best -> worst
f = f.sort_values(by=[sort_col, "RS 1Y", "RS 6M", "RS 3M", "RS 1M", "RS 1W"], ascending=False).copy()

# Limit results
f = f.head(int(top_n)).copy()

# =========================
# ADD PRICE + NAME + OPTIONAL SPARK
# =========================
# Last price
last_prices = prices.iloc[-1].to_dict()
f["Price"] = f["Ticker"].map(lambda t: float(last_prices.get(t, np.nan)))

# Names only for results (fast)
result_tickers = f["Ticker"].tolist()
name_map = fetch_names_for_results(result_tickers)
f["Name"] = f["Ticker"].map(lambda t: name_map.get(t, t))

# RS sparkline (1M ratio line, same concept as dashboard)
# We build it for results only.
sparks = []
levels_list = []
b = prices[benchmark]
for t in result_tickers:
    if t not in prices.columns:
        sparks.append("")
        levels_list.append([])
        continue
    s = prices[t]
    rs_ratio_series = (s / s.shift(21)) / (b / b.shift(21))
    spark, levels = sparkline_from_series(rs_ratio_series, n=26)
    sparks.append(spark)
    levels_list.append(levels)

f["RS Spark"] = sparks
f["__spark_levels"] = levels_list

# Display columns
display_cols = ["Ticker", "Name", "Price", "RS Spark", "RS Trend", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]

# =========================
# RESULTS
# =========================
st.markdown('<div class="section-title">Scan Results</div>', unsafe_allow_html=True)

st.caption(
    f"Matches: {len(f):,} • Universe scanned: {len(universe):,} • "
    f"RS = relative to {benchmark}, percentile-ranked (1–99) vs your universe."
)

if f.empty:
    st.warning("No matches. Lower your minimum RS filters or change Scan Mode.")
    st.stop()

render_table_html(f, display_cols, height_px=820)

# =========================
# EXPORT
# =========================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

export_df = f[["Ticker", "Name", "Price", "RS Trend", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]].copy()
export_df.insert(0, "Benchmark", benchmark)

csv_out = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_out, file_name="rs_scanner_results.csv", mime="text/csv")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    """
**How RS Ratings are computed in this Scanner**

- **Relative Performance vs Benchmark:** For each ticker and each timeframe (1W/1M/3M/6M/1Y), we measure outperformance vs the benchmark (default SPY).
- **1–99 Rating:** For each timeframe, we percentile-rank those relative performance values across **your full universe** (Tickers.csv) and scale to **1–99**.
- This means RS=99 represents the top ~1% of tickers in your scanned universe for that timeframe.
"""
)

