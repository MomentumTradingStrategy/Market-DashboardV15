import os
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Scanner (yfinance)", layout="wide")

CSV_FILE = "Tickers.csv"
BENCHMARK = "SPY"

# How far back to pull prices (2y is plenty for RS 1Y=252 trading days)
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

# Local cache (persists on Streamlit Cloud disk between runs until redeploy/sleep)
CACHE_PARQUET = "yf_prices_cache.parquet"
CACHE_META = "yf_cache_meta.txt"


# ============================================================
# HELPERS
# ============================================================
def utc_now_label():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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
    ticks = list(dict.fromkeys(ticks))  # preserve order, unique
    return ticks


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def save_cache(df_prices: pd.DataFrame) -> None:
    df_prices.to_parquet(CACHE_PARQUET)
    with open(CACHE_META, "w") as f:
        f.write(f"Saved: {utc_now_label()} | rows={len(df_prices)} cols={len(df_prices.columns)}\n")


def load_cache() -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(CACHE_PARQUET):
        return pd.DataFrame(), "No local cache found."
    try:
        df = pd.read_parquet(CACHE_PARQUET)
        meta = ""
        if os.path.exists(CACHE_META):
            with open(CACHE_META, "r") as f:
                meta = f.read().strip()
        return df, (meta or "Loaded local cache.")
    except Exception as e:
        return pd.DataFrame(), f"Failed to load cache: {e}"


def _extract_close_matrix(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    yfinance download returns different shapes depending on count:
    - MultiIndex columns (field, ticker) or (ticker, field)
    - or single-index columns for 1 ticker
    We return a DataFrame with columns=tickers and values=Close (auto_adjusted).
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # If 1 ticker, columns like: Open High Low Close Volume
    if isinstance(raw.columns, pd.Index) and "Close" in raw.columns:
        # single ticker case -> name it
        t = tickers[0]
        out = raw[["Close"]].rename(columns={"Close": t})
        return out

    # MultiIndex case
    if isinstance(raw.columns, pd.MultiIndex):
        # Could be (PriceField, Ticker) or (Ticker, PriceField)
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))

        if "Close" in lvl0:
            # (Field, Ticker)
            close = raw["Close"]
            # close columns are tickers
            return close

        if "Close" in lvl1:
            # (Ticker, Field)
            close = raw.xs("Close", axis=1, level=1)
            return close

    # Fallback: try common patterns
    for col in ["Adj Close", "Close"]:
        try:
            if col in raw.columns:
                return raw[[col]].rename(columns={col: tickers[0]})
        except Exception:
            pass

    return pd.DataFrame()


def rs_ratio(close_t: pd.Series, close_b: pd.Series, periods: int) -> float:
    """
    Relative performance ratio of ticker vs benchmark over 'periods' trading days.
    Returns last value of ( (t/t_shift) / (b/b_shift) ) - 1
    """
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    rr = (t / b) - 1
    rr = rr.dropna()
    if rr.empty:
        return np.nan
    return float(rr.iloc[-1])


# ============================================================
# DATA PULL (MANUAL REFRESH) — cached by refresh_id
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_yf_chunked(
    tickers: Tuple[str, ...],
    batch_size: int,
    refresh_id: int,
    period: str = PRICE_PERIOD,
    interval: str = INTERVAL
) -> pd.DataFrame:
    """
    refresh_id is included to bust cache only when user clicks Refresh.
    """
    ticks = list(tickers)
    batches = chunk_list(ticks, batch_size)

    all_close = []
    for batch in batches:
        # yf.download handles multiple tickers efficiently
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
        if close is not None and not close.empty:
            all_close.append(close)

    if not all_close:
        return pd.DataFrame()

    df = pd.concat(all_close, axis=1)

    # Ensure unique columns and normalized tickers
    df.columns = [normalize_ticker(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Sort by date and forward-fill gaps
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df.sort_index().ffill()

    # Use float32 to reduce memory
    df = df.astype("float32")

    return df


# ============================================================
# UI
# ============================================================
st.title("Relative Strength Scanner (yfinance • Manual Refresh)")
st.caption(f"As of: {utc_now_label()} • Benchmark: {BENCHMARK}")

# Session state for manual refresh cache-bust
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
    st.subheader("Data")
    batch_size = st.slider("Batch size (tickers per request)", 100, 800, 400, step=50)
    refresh = st.button("Refresh data now", use_container_width=True)

    st.caption("Tip: If Yahoo throttles, lower batch size (e.g., 250–350).")

    st.divider()
    st.subheader("Scanner Controls")
    rs_min = st.slider("Minimum RS Rating", min_value=1, max_value=99, value=90, step=1)

    primary_tf = st.selectbox(
        "Primary Timeframe",
        ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"],
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

# If user clicks refresh, bump refresh_id to bust st.cache_data
if refresh:
    st.session_state.refresh_id += 1

# Load cached prices from disk first (so you always have something)
disk_df, disk_meta = load_cache()
if not disk_df.empty:
    st.info(f"Using saved cache until refresh completes. ({disk_meta})")

# If refresh was clicked, fetch fresh data and save it
if refresh:
    st.warning("Refreshing prices from Yahoo Finance… (this may take a bit for 6,600 tickers)")
    prog = st.progress(0)
    status = st.empty()

    # We can’t hook into yf.download progress easily, so we do a simple visual stepper
    # by estimating number of batches
    batches = chunk_list(universe, batch_size)
    total_batches = len(batches)

    # Fetch in one cached function call (fastest), but show a rough progress bar
    # Update progress “optimistically” in a loop so UI isn’t frozen
    # (We still rely on the function doing the actual work.)
    try:
        # Kick the work
        status.write(f"Downloading {len(universe):,} tickers in {total_batches} batches…")
        df_prices = fetch_prices_yf_chunked(
            tickers=tuple(universe),
            batch_size=batch_size,
            refresh_id=st.session_state.refresh_id,
            period=PRICE_PERIOD,
            interval=INTERVAL,
        )
        prog.progress(1.0)

        if df_prices.empty:
            raise RuntimeError("Yahoo returned empty data. Try smaller batch size.")

        # Save to disk cache
        save_cache(df_prices)
        status.success(f"Refresh complete. Saved cache at {utc_now_label()}.")
        disk_df = df_prices

    except Exception as e:
        status.error(f"Refresh failed: {e}")
        if disk_df.empty:
            st.stop()
        st.warning("Falling back to last saved cache.")

# Use whatever we have (fresh or disk)
price_df = disk_df.copy()
if price_df.empty:
    st.error("No data available. Click Refresh data now.")
    st.stop()

# Validate benchmark
if bench_norm not in price_df.columns:
    st.error(f"Benchmark {BENCHMARK} missing from downloaded data. Try refresh again.")
    st.stop()

# Ensure enough history for 1Y
min_len_needed = max(HORIZONS.values()) + 5
usable_cols = [c for c in price_df.columns if price_df[c].dropna().shape[0] >= min_len_needed]
if bench_norm not in usable_cols:
    st.error(f"Benchmark {BENCHMARK} doesn’t have enough history. Try refresh again.")
    st.stop()

price_df = price_df[usable_cols].copy()
bench = price_df[bench_norm]

st.write(f"Tickers with sufficient history (including benchmark): **{len(price_df.columns):,}**")

# Compute RS ratios
rows = []
tickers_to_score = [t for t in price_df.columns if t != bench_norm]

for t in tickers_to_score:
    close = price_df[t]
    rec = {"Ticker": t}
    for col, n in HORIZONS.items():
        rec[col] = rs_ratio(close, bench, n)
    rows.append(rec)

df = pd.DataFrame(rows)

# Convert each RS column to 1–99 percentile across universe
for col in HORIZONS.keys():
    s = pd.to_numeric(df[col], errors="coerce")
    df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

# Apply scan filters
if mode == "Primary timeframe only":
    df_f = df[df[primary_tf] >= rs_min].copy()

elif mode == "All timeframes >= threshold":
    cond = True
    for col in HORIZONS.keys():
        cond = cond & (df[col] >= rs_min)
    df_f = df[cond].copy()

elif mode == "Accelerating (RS 1Y → RS 1W improving)":
    df_f = df[(df["RS 1W"] > df["RS 1Y"]) & (df[primary_tf] >= rs_min)].copy()

else:  # Decelerating
    df_f = df[(df["RS 1W"] < df["RS 1Y"]) & (df[primary_tf] >= rs_min)].copy()

df_f = df_f.sort_values([primary_tf, "RS 1Y"], ascending=[False, False])

st.success(f"Scan complete • {len(df_f):,} matches")

df_show = df_f.head(max_results).reset_index(drop=True)
st.dataframe(df_show, use_container_width=True, height=720)

st.markdown(
    """
**How ranking works**  
- Each stock is compared to **SPY** over each timeframe (1W/1M/3M/6M/1Y).  
- Those relative-performance values are percentile-ranked **across your universe** into **RS 1–99**.  
- RS 99 ≈ “top ~1% of your list” for that timeframe.
"""
)

with st.expander("Troubleshooting / Tuning"):
    st.write(
        "- If refresh fails or is slow: lower **batch size** (try 250–350).\n"
        "- Yahoo can throttle randomly. This app keeps the last saved cache so you don’t lose the dashboard.\n"
        "- If a few symbols don’t return data, that’s normal with Yahoo; they’ll be excluded from scoring.\n"
        "- Want it even faster? Set PRICE_PERIOD to '1y' once you’re comfortable (still enough for RS 1Y if trading days are present).\n"
    )


