from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# Page
# =========================
st.set_page_config(page_title="Relative Strength Scanner", layout="wide")


def _asof_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# =========================
# Styling (keep it clean + readable)
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.card h3{margin:0 0 6px 0; font-size: 1.05rem; font-weight: 950;}
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
td.ticker {font-weight: 950;}
td.mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# Config
# =========================
DEFAULT_TICKERS_FILE = "tickers.csv"
DEFAULT_BENCHMARK = "SPY"

# Trading-day approximations (same as your RS dashboard logic)
HORIZONS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


# =========================
# Helpers
# =========================
def _clean_symbol(x: str) -> str:
    x = str(x).strip().upper()
    if not x:
        return ""
    # a couple common cleanup patterns
    x = x.replace("\u200b", "")  # zero-width spaces
    return x


def load_tickers_from_csv_path(path: str) -> list[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    # Take the first non-empty column
    col = df.columns[0]
    tickers = [_clean_symbol(v) for v in df[col].tolist()]
    tickers = [t for t in tickers if t and t != "SYMBOL" and t != "TICKER"]
    # De-dupe, keep order
    seen = set()
    out = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_tickers_from_uploaded(file) -> list[str]:
    df = pd.read_csv(file)
    if df.empty:
        return []
    col = df.columns[0]
    tickers = [_clean_symbol(v) for v in df[col].tolist()]
    tickers = [t for t in tickers if t and t != "SYMBOL" and t != "TICKER"]
    seen = set()
    out = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def pct_change(close: pd.Series, periods: int) -> pd.Series:
    return close.pct_change(periods=periods)


def rs_ratio_return(close_t: pd.Series, close_b: pd.Series, periods: int) -> pd.Series:
    """
    RS over a window = % change of (ticker / benchmark) over that window.
    Equivalent to: ( (t / t[-n]) / (b / b[-n]) ) - 1
    """
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1


def rs_rank_1_99(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    # rank percent -> 1..99
    return (s.rank(pct=True) * 99).round().clip(1, 99)


def render_table_html(df: pd.DataFrame, height_px: int = 800):
    columns = list(df.columns)

    th = "".join([f"<th>{c}</th>" for c in columns])
    trs = []
    for _, row in df.iterrows():
        tds = []
        for c in columns:
            v = row.get(c, "")
            cls = ""
            if c == "Ticker":
                cls = "ticker"
            if c in ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y", "Accel", "Decel"]:
                cls = "mono"
            tds.append(f'<td class="{cls}">{"" if pd.isna(v) else v}</td>')
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
# Price fetching (chunked)
# =========================
@dataclass
class FetchResult:
    close: pd.DataFrame  # columns=tickers, index=date
    missing: list[str]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_closes_chunked(
    tickers: list[str],
    period: str,
    chunk_size: int,
    pause_s: float,
) -> FetchResult:
    """
    Pull adjusted daily closes for a large universe in chunks using yfinance.

    Notes:
    - yfinance can be flaky. Chunking + small pause helps.
    - This returns a wide close dataframe with columns for each ticker found.
    """
    all_close_parts = []
    missing = []

    # yfinance sometimes fails on bad symbols; we keep going
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            df = yf.download(
                tickers=chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            df = pd.DataFrame()

        if df is None or df.empty:
            # mark all missing for this chunk
            missing.extend(chunk)
            time.sleep(pause_s)
            continue

        # MultiIndex columns if multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            closes = {}
            for t in chunk:
                if (t, "Close") in df.columns:
                    closes[t] = df[(t, "Close")]
            close_df = pd.DataFrame(closes)
        else:
            # single ticker
            close_df = pd.DataFrame({chunk[0]: df["Close"]})

        # Track missing inside chunk
        found = set(close_df.columns.tolist())
        for t in chunk:
            if t not in found:
                missing.append(t)

        all_close_parts.append(close_df)
        time.sleep(pause_s)

    if not all_close_parts:
        return FetchResult(close=pd.DataFrame(), missing=sorted(list(set(missing))))

    close_all = pd.concat(all_close_parts, axis=1)
    close_all = close_all.dropna(how="all").ffill()
    # De-dupe columns if any repeats
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]

    return FetchResult(close=close_all, missing=sorted(list(set(missing))))


# =========================
# UI
# =========================
st.title("Relative Strength Scanner")
st.caption(f"As of: {_asof_ts()} • Benchmark: {DEFAULT_BENCHMARK}")

with st.sidebar:
    st.subheader("Universe")
    use_repo_csv = st.checkbox(f"Use repo file: {DEFAULT_TICKERS_FILE}", value=True)

    uploaded = None
    if not use_repo_csv:
        uploaded = st.file_uploader("Upload tickers CSV", type=["csv"])

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.subheader("Scan Settings")
    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK).strip().upper() or DEFAULT_BENCHMARK
    period = st.selectbox(
        "History pulled (more history = slower)",
        options=["18mo", "2y", "3y"],
        index=1,
    )

    chunk_size = st.slider("Batch size", min_value=50, max_value=400, value=200, step=25)
    pause_s = st.slider("Pause between batches (seconds)", min_value=0.0, max_value=2.0, value=0.25, step=0.05)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.subheader("Filters")
    mode = st.selectbox(
        "Scan mode",
        options=[
            "RS threshold (single timeframe)",
            "RS threshold (all selected timeframes)",
            "Accelerating leaders (improving RS)",
            "Decelerating leaders (weakening RS)",
        ],
        index=0,
    )

    tf_sort = st.selectbox("Sort timeframe", options=list(HORIZONS.keys()), index=1)  # default 1M
    min_rs = st.slider("Minimum RS (1–99)", min_value=1, max_value=99, value=90, step=1)
    max_rows = st.slider("Max results to show", min_value=50, max_value=1000, value=250, step=50)

    tfs = st.multiselect(
        "Timeframes used for filter (where applicable)",
        options=list(HORIZONS.keys()),
        default=["1W", "1M", "3M", "6M", "1Y"],
    )

    run = st.button("Run Scan", type="primary", use_container_width=True)

# Load tickers
tickers: list[str] = []
tickers_source = ""

if use_repo_csv:
    if os.path.exists(DEFAULT_TICKERS_FILE):
        tickers = load_tickers_from_csv_path(DEFAULT_TICKERS_FILE)
        tickers_source = f"Loaded from repo file: {DEFAULT_TICKERS_FILE}"
    else:
        tickers_source = f"Could not find {DEFAULT_TICKERS_FILE}


