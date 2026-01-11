import os
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Scanner (yfinance)", layout="wide")

CSV_FILE = "Tickers.csv"
BENCHMARK = "SPY"

# Pull enough history for RS 1Y (252 trading days) + buffer
PRICE_PERIOD = "2y"
INTERVAL = "1d"

# RS horizons (trading days)
HORIZONS = {
    "RS 1W": 5,
    "RS 1M": 21,
    "RS 3M": 63,
    "RS 6M": 126,
    "RS 1Y": 252,
}

# Cache files (Streamlit Cloud file system persists between reruns; may reset on redeploy)
CACHE_PARQUET = "yf_prices_cache.parquet"
CACHE_META_TXT = "yf_cache_meta.txt"


# ============================================================
# HELPERS
# ============================================================
def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


SPECIAL_TICKER_MAP = {
    "BRK-A": "BRK.BRK-A",  # (we override below; keep map simple)
}

# Better: explicit mappings for class shares
SPECIAL_TICKER_MAP = {
    "BRK-A": "BRK.A",
    "BRK-B": "BRK.B",
    "BRKA": "BRK.A",
    "BRKB": "BRK.B",
}


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = t.replace(" ", "")
    t = t.replace("/", "-")
    t = SPECIAL_TICKER_MAP.get(t, t)
    return t


def load_universe(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    col = None
    for c in df.columns:
        if c.strip().lower() in ("ticker", "symbol"):
            col = c
            break
    if col is None:
        col = df.columns[0]

    ticks = (
        df[col]
        .astype(str)
        .map(normalize_ticker)
        .replace({"NAN": "", "NONE": ""})
        .tolist()
    )
    ticks = [t for t in ticks if t and t.isascii()]
    ticks = list(dict.fromkeys(ticks))  # unique preserve order
    return ticks


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def save_cache(df_prices: pd.DataFrame) -> None:
    df_prices.to_parquet(CACHE_PARQUET, index=True)
    with open(CACHE_META_TXT, "w") as f:
        f.write(f"Saved: {utc_now_label()} | rows={len(df_prices):,} cols={len(df_prices.columns):,}\n")


def load_cache() -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(CACHE_PARQUET):
        return pd.DataFrame(), "No cache file found."
    try:
        df = pd.read_parquet(CACHE_PARQUET)
        meta = ""
        if os.path.exists(CACHE_META_TXT):
            with open(CACHE_META_TXT, "r") as f:
                meta = f.read().strip()
        return df, (meta or "Loaded cache.")
    except Exception as e:
        return pd.DataFrame(), f"Failed to load cache: {e}"


def _extract_close_matrix(raw: pd.DataFrame, tickers_in_batch: List[str]) -> pd.DataFrame:
    """
    yfinance download output:
      - single ticker: columns like Open/High/Low/Close/Volume
      - multi ticker: MultiIndex columns; either (PriceField, Ticker) or (Ticker, PriceField)
    We return Close matrix with columns = tickers.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Single ticker case
    if isinstance(raw.columns, pd.Index) and "Close" in raw.columns:
        t = normalize_ticker(tickers_in_batch[0])
        out = raw[["Close"]].rename(columns={"Close": t})
        return out

    # MultiIndex case
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))

        if "Close" in lvl0:
            # (Field, Ticker)
            close = raw["Close"].copy()
            close.columns = [normalize_ticker(c) for c in close.columns]
            return close

        if "Close" in lvl1:
            # (Ticker, Field)
            close = raw.xs("Close", axis=1, level=1).copy()
            close.columns = [normalize_ticker(c) for c in close.columns]
            return close

    return pd.DataFrame()


def rs_ratio(close_t: pd.Series, close_b: pd.Series, periods: int) -> float:
    """
    ( (t/t_shift) / (b/b_shift) ) - 1
    """
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    rr = (t / b) - 1
    rr = rr.dropna()
    if rr.empty:
        return np.nan
    return float(rr.iloc[-1])


# ============================================================
# DATA DOWNLOAD (MANUAL) — uses refresh_id to bust cache
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_yf_chunked(
    tickers: Tuple[str, ...],
    batch_size: int,
    refresh_id: int,
    period: str,
    interval: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (close_matrix_df, failed_tickers)
    refresh_id ensures this runs only when user clicks Refresh.
    """
    ticks = [normalize_ticker(t) for t in tickers]
    ticks = list(dict.fromkeys([t for t in ticks if t]))

    failed: List[str] = []
    closes: List[pd.DataFrame] = []

    # yfinance can be sensitive; smaller batches are more reliable for huge universes
    for batch in chunk_list(ticks, batch_size):
        try:
            raw = yf.download(
                tickers=" ".join(batch),
                period=period,
                interval=interval,
                group_by="column",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            close = _extract_close_matrix(raw, batch)
            if close is None or close.empty:
                failed.extend(batch)
            else:
                closes.append(close)
        except Exception:
            failed.extend(batch)

    if not closes:
        return pd.DataFrame(), failed

    df = pd.concat(closes, axis=1)

    # Normalize index
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").normalize()
    df = df.sort_index()

    # De-dupe columns
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Forward-fill gaps
    df = df.ffill()

    # Reduce memory footprint (6600 cols can be heavy)
    df = df.astype("float32")

    return df, failed


# ============================================================
# UI
# ============================================================
st.title("Relative Strength Scanner (yfinance • Manual Refresh)")
st.caption(f"As of: {utc_now_label()} • Benchmark: {BENCHMARK}")

# session state
if "refresh_id" not in st.session_state:
    st.session_state.refresh_id = 0

# Load tickers
try:
    universe = load_universe(CSV_FILE)
except Exception as e:
    st.error(f"Could not load {CSV_FILE}: {e}")
    st.stop()

if not universe:
    st.error(f"{CSV_FILE} loaded, but no tickers were found.")
    st.stop()

bench_norm = normalize_ticker(BENCHMARK)
if bench_norm not in universe:
    universe = universe + [bench_norm]

st.write(f"Universe size: **{len(universe):,}** tickers (including benchmark).")

with st.sidebar:
    st.subheader("Data Controls")

    batch_size = st.slider(
        "Batch size (tickers per request)",
        min_value=100,
        max_value=800,
        value=300,
        step=50
    )

    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d"], index=0)

    refresh_btn = st.button("Refresh data now", use_container_width=True)

    st.caption("If refresh fails: lower batch size (250–350) and try again.")

    st.divider()
    st.subheader("Scanner Controls")

    rs_min = st.slider("Minimum RS Rating", 1, 99, 90, 1)

    primary_tf = st.selectbox(
        "Primary Timeframe",
        list(HORIZONS.keys()),
        index=1
    )

    mode = st.selectbox(
        "Scan Mode",
        [
            "Primary timeframe only",
            "All timeframes >= threshold",
            "Accelerating (RS 1Y → RS 1W improving)",
            "Decelerating (RS 1W → RS 1Y weakening)",
        ],
        index=0
    )

    max_results = st.slider("Max Results to Display", 25, 500, 200, step=25)


# Always try to load disk cache first so app never goes blank
cache_df, cache_meta = load_cache()
if not cache_df.empty:
    st.info(f"Loaded last saved cache. {cache_meta}")

# Manual refresh
failed_total: List[str] = []
if refresh_btn:
    st.session_state.refresh_id += 1
    st.warning("Refreshing prices from Yahoo Finance… (large universes can take time)")
    df_new, failed_total = fetch_prices_yf_chunked(
        tickers=tuple(universe),
        batch_size=batch_size,
        refresh_id=st.session_state.refresh_id,
        period=period,
        interval=interval,
    )
    if df_new.empty:
        st.error("Yahoo returned empty data. Lower batch size and try again (e.g., 250–300).")
        if cache_df.empty:
            st.stop()
        st.warning("Using last saved cache instead.")
    else:
        save_cache(df_new)
        cache_df = df_new
        st.success(f"Refresh complete. Cached {len(cache_df.columns):,} tickers.")

# Use cache_df
price_df = cache_df.copy()
if price_df.empty:
    st.error("No cache yet. Click **Refresh data now**.")
    st.stop()

# Ensure benchmark exists
if bench_norm not in price_df.columns:
    st.error(f"Benchmark {BENCHMARK} missing from data. Try refresh again.")
    st.stop()

# Ensure enough history for 1Y
min_len_needed = max(HORIZONS.values()) + 5
usable_cols = [c for c in price_df.columns if price_df[c].dropna().shape[0] >= min_len_needed]
if bench_norm not in usable_cols:
    st.error(f"Benchmark {BENCHMARK} doesn’t have enough history yet. Try refresh again.")
    st.stop()

price_df = price_df[usable_cols].copy()
bench = price_df[bench_norm]

st.write(f"Tickers with sufficient history (including benchmark): **{len(price_df.columns):,}**")

if failed_total:
    with st.expander("Download issues (some tickers may be missing)"):
        st.write(f"Failed tickers count: {len(failed_total):,}")
        st.write(", ".join(failed_total[:500]) + (" ..." if len(failed_total) > 500 else ""))

# Compute RS
tickers_to_score = [t for t in price_df.columns if t != bench_norm]

rows = []
for t in tickers_to_score:
    close = price_df[t]
    rec = {"Ticker": t}
    for col, n in HORIZONS.items():
        rec[col] = rs_ratio(close, bench, n)
    rows.append(rec)

df = pd.DataFrame(rows)

# Percentile rank -> 1..99
for col in HORIZONS.keys():
    s = pd.to_numeric(df[col], errors="coerce")
    df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

# Filter
if mode == "Primary timeframe only":
    df_f = df[df[primary_tf] >= rs_min].copy()
elif mode == "All timeframes >= threshold":
    cond = True
    for col in HORIZONS.keys():
        cond = cond & (df[col] >= rs_min)
    df_f = df[cond].copy()
elif mode == "Accelerating (RS 1Y → RS 1W improving)":
    df_f = df[(df["RS 1W"] > df["RS 1Y"]) & (df[primary_tf] >= rs_min)].copy()
else:
    df_f = df[(df["RS 1W"] < df["RS 1Y"]) & (df[primary_tf] >= rs_min)].copy()

df_f = df_f.sort_values([primary_tf, "RS 1Y"], ascending=[False, False])

st.success(f"Scan complete • {len(df_f):,} matches")
st.dataframe(df_f.head(max_results).reset_index(drop=True), use_container_width=True, height=720)

st.markdown(
    """
**How ranking works**  
- Each stock is compared to **SPY** over each timeframe (1W/1M/3M/6M/1Y).  
- Those relative-performance values are percentile-ranked across your universe into **RS 1–99**.  
"""
)

with st.expander("If refresh is slow / fails"):
    st.write(
        "- Lower **batch size** to 250–300.\n"
        "- Use **2y** history for reliable 1Y RS.\n"
        "- Yahoo throttles sometimes; this app keeps your last saved cache so you can still scan.\n"
    )



