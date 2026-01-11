import os
import time
import json
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Scanner (Alpha Vantage Cache)", layout="wide")

CSV_FILE = "Tickers.csv"
BENCHMARK = "SPY"

# Alpha Vantage rate limit (free is usually 5/min). If you have a paid plan, raise it.
CALLS_PER_MINUTE = 5

# How many trading days we keep per ticker (enough for RS 1Y=252 plus buffer)
KEEP_TRADING_DAYS = 320

# Cache files (Streamlit Cloud: ephemeral but persists across runs until redeploy / sleep)
CACHE_PARQUET = "av_prices_cache.parquet"     # long format: date, ticker, close
META_JSON = "av_cache_meta.json"              # { "TICKER": {"updated_utc": "..."} }

# RS horizons (trading days)
HORIZONS = {
    "RS 1W": 5,
    "RS 1M": 21,
    "RS 3M": 63,
    "RS 6M": 126,
    "RS 1Y": 252,
}


# ============================================================
# API KEY
# ============================================================
def get_api_key() -> str:
    try:
        k = st.secrets.get("AlphaVantage_API_KEY", "")
        if k:
            return str(k).strip()
    except Exception:
        pass
    return os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()


API_KEY = get_api_key()


def utc_now_label():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================
# Ticker normalization for Alpha Vantage
# ============================================================
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


def load_universe(csv_path: str) -> list[str]:
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


# ============================================================
# Rate limiting (global)
# ============================================================
_min_interval = 60.0 / max(1, int(CALLS_PER_MINUTE))
_rate_lock = threading.Lock()
_last_call_ts = 0.0

def _rate_limit_wait():
    global _last_call_ts
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_ts
        wait = _min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()


# ============================================================
# Alpha Vantage fetch (Daily Adjusted)
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)  # cache each ticker response 12h
def fetch_daily_adj_closes_av(ticker: str) -> pd.Series:
    """
    Returns a Series indexed by UTC-normalized date with adjusted close.
    Keeps full output from AV, but we later trim to KEEP_TRADING_DAYS.
    """
    if not API_KEY:
        raise RuntimeError("Missing AlphaVantage_API_KEY.")

    _rate_limit_wait()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "Note" in js:
        # Rate-limit response
        raise RuntimeError(js["Note"])
    if "Error Message" in js:
        raise RuntimeError(js["Error Message"])

    ts_key = "Time Series (Daily)"
    if ts_key not in js:
        raise RuntimeError(f"Unexpected Alpha Vantage response keys for {ticker}: {list(js.keys())[:6]}")

    rows = js[ts_key]
    if not rows:
        return pd.Series(dtype="float64", name=ticker)

    idx, vals = [], []
    for d, rec in rows.items():
        v = rec.get("5. adjusted close")
        if v is None:
            continue
        try:
            idx.append(pd.to_datetime(d, utc=True).normalize())
            vals.append(float(v))
        except Exception:
            continue

    if not idx:
        return pd.Series(dtype="float64", name=ticker)

    s = pd.Series(vals, index=idx, name=ticker).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ============================================================
# Disk cache (prices + metadata)
# ============================================================
def load_meta() -> dict:
    if not os.path.exists(META_JSON):
        return {}
    try:
        with open(META_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_meta(meta: dict) -> None:
    with open(META_JSON, "w") as f:
        json.dump(meta, f)

def load_price_cache_long() -> pd.DataFrame:
    """
    Long format: date, ticker, close
    """
    if not os.path.exists(CACHE_PARQUET):
        return pd.DataFrame(columns=["date", "ticker", "close"])
    try:
        df = pd.read_parquet(CACHE_PARQUET)
        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "close"])
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
        return df[["date", "ticker", "close"]]
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close"])

def save_price_cache_long(df_long: pd.DataFrame) -> None:
    # Keep it compact and consistent
    df_long = df_long.copy()
    df_long["date"] = pd.to_datetime(df_long["date"], utc=True).dt.normalize()
    df_long["ticker"] = df_long["ticker"].astype(str)
    df_long["close"] = pd.to_numeric(df_long["close"], errors="coerce")
    df_long = df_long.dropna(subset=["date", "ticker", "close"])
    df_long.to_parquet(CACHE_PARQUET, index=False)

def upsert_ticker_into_cache(df_long: pd.DataFrame, ticker: str, ser: pd.Series) -> pd.DataFrame:
    """
    Replace existing ticker rows with new trimmed data.
    """
    ticker = normalize_ticker(ticker)
    ser = ser.dropna().sort_index()

    # Trim to last KEEP_TRADING_DAYS trading rows
    ser = ser.tail(KEEP_TRADING_DAYS)

    new_rows = pd.DataFrame({
        "date": ser.index,
        "ticker": ticker,
        "close": ser.values.astype(float),
    })

    if df_long.empty:
        return new_rows

    df_long = df_long[df_long["ticker"] != ticker]
    return pd.concat([df_long, new_rows], ignore_index=True)


# ============================================================
# RS math
# ============================================================
def rs_ratio(close_t: pd.Series, close_b: pd.Series, periods: int) -> float:
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    rr = (t / b) - 1
    rr = rr.dropna()
    if rr.empty:
        return np.nan
    return float(rr.iloc[-1])


def build_price_matrix_from_cache(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot cache to wide date-index DataFrame, ffill missing.
    """
    if df_long.empty:
        return pd.DataFrame()

    dfp = df_long.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    dfp = dfp.ffill()
    return dfp


# ============================================================
# UI
# ============================================================
st.title("Relative Strength Scanner (Alpha Vantage • Cached)")
st.caption(f"As of: {utc_now_label()} • Benchmark: {BENCHMARK}")

if not API_KEY:
    st.error(
        "Missing API key.\n\n"
        "Set **AlphaVantage_API_KEY** in Streamlit Secrets (recommended) "
        "or set env var **ALPHAVANTAGE_API_KEY**."
    )
    st.stop()

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

# Load cache
meta = load_meta()
cache_long = load_price_cache_long()

cached_tickers = sorted(cache_long["ticker"].unique().tolist()) if not cache_long.empty else []
st.write(f"Cached tickers: **{len(cached_tickers):,}**")

with st.sidebar:
    st.subheader("Cache Builder (for big universes)")

    batch_size = st.slider("Update this many tickers per click", 10, 300, 60, step=10)
    prefer_uncached = st.checkbox("Prioritize uncached tickers first", value=True)

    st.caption(f"Rate limit set to ~{CALLS_PER_MINUTE}/min. (Change CALLS_PER_MINUTE in code if your AV plan allows more.)")

    update_cache_btn = st.button("Update cache (batch)", use_container_width=True)

    st.divider()
    st.subheader("Scanner Controls")

    rs_min = st.slider("Minimum RS Rating", min_value=1, max_value=99, value=90, step=1)
    primary_tf = st.selectbox("Primary Timeframe", list(HORIZONS.keys()), index=1)

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
    run_scan_btn = st.button("Run Scan", use_container_width=True)


# ----------------------------
# Cache updater
# ----------------------------
if update_cache_btn:
    st.info("Updating cache… this is rate-limited. Click again later to continue building the full universe.")

    # Choose tickers to update
    universe_norm = [normalize_ticker(t) for t in universe]
    universe_norm = list(dict.fromkeys([t for t in universe_norm if t]))

    cached_set = set(cached_tickers)

    if prefer_uncached:
        candidates = [t for t in universe_norm if t not in cached_set]
        # If everything cached, refresh oldest updated
        if not candidates:
            # sort by meta updated time (oldest first)
            def upd_ts(t):
                v = meta.get(t, {}).get("updated_utc", "")
                return v or "0000-00-00 00:00 UTC"
            candidates = sorted(universe_norm, key=upd_ts)
    else:
        # refresh oldest first across all
        def upd_ts(t):
            v = meta.get(t, {}).get("updated_utc", "")
            return v or "0000-00-00 00:00 UTC"
        candidates = sorted(universe_norm, key=upd_ts)

    todo = candidates[:batch_size]
    st.write(f"Updating **{len(todo)}** tickers…")

    prog = st.progress(0)
    status = st.empty()

    failures = []
    for i, t in enumerate(todo, start=1):
        status.write(f"Downloading {t} ({i}/{len(todo)}) …")
        try:
            ser = fetch_daily_adj_closes_av(t)
            if ser is None or ser.empty:
                failures.append(t)
            else:
                cache_long = upsert_ticker_into_cache(cache_long, t, ser)
                meta[t] = {"updated_utc": utc_now_label()}
        except Exception:
            failures.append(t)

        prog.progress(i / len(todo))

    # Save to disk
    save_price_cache_long(cache_long)
    save_meta(meta)

    st.success(
        f"Cache update complete. Added/updated: {len(todo) - len(failures)} • Failed: {len(failures)}"
    )

    if failures:
        with st.expander("Failed tickers"):
            st.write(", ".join(failures[:500]))
            if len(failures) > 500:
                st.write(f"...and {len(failures) - 500} more")

    st.rerun()


# ----------------------------
# Run scan from cache
# ----------------------------
if run_scan_btn:
    if cache_long.empty:
        st.error("Cache is empty. Click **Update cache (batch)** first.")
        st.stop()

    price_df = build_price_matrix_from_cache(cache_long)

    if bench_norm not in price_df.columns:
        st.error(f"Benchmark {BENCHMARK} not found in cache yet. Update cache until it includes {bench_norm}.")
        st.stop()

    # Keep only tickers with enough history for 1Y
    min_len_needed = max(HORIZONS.values()) + 5
    usable_cols = []
    for c in price_df.columns:
        if price_df[c].dropna().shape[0] >= min_len_needed:
            usable_cols.append(c)

    if bench_norm not in usable_cols:
        st.error(f"Benchmark {BENCHMARK} does not have enough history yet in cache.")
        st.stop()

    price_df = price_df[usable_cols].copy()
    bench = price_df[bench_norm]

    tickers_to_score = [t for t in price_df.columns if t != bench_norm]

    if len(tickers_to_score) < 20:
        st.warning("Your cache is still small. Results will improve as you build more tickers into the cache.")

    rows = []
    for t in tickers_to_score:
        close = price_df[t]
        rec = {"Ticker": t}
        for col, n in HORIZONS.items():
            rec[col] = rs_ratio(close, bench, n)
        rows.append(rec)

    df = pd.DataFrame(rows)

    # percentile -> 1..99 across cached universe (NOT the full 6600 yet)
    for col in HORIZONS.keys():
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

    # filters
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
    df_show = df_f.head(max_results).reset_index(drop=True)

    st.success(f"Scan complete • Cached universe scored: {len(df):,} • Matches: {len(df_f):,}")
    st.dataframe(df_show, use_container_width=True, height=700)

    st.markdown(
        """
**Important**  
- RS 1–99 is computed across your **cached tickers**, not your full 6,600, until the cache is fully built.  
- Click **Update cache (batch)** repeatedly over time to grow coverage.  
- Free Alpha Vantage will take many hours to fully populate 6,600 tickers.
"""
    )

