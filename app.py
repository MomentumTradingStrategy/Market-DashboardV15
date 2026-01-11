import os
import time
import threading
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Scanner", layout="wide")

CSV_FILE = "Tickers.csv"
BENCHMARK = "SPY"

# Alpha Vantage free tier is typically 5 calls/minute.
# If you have a higher tier, raise this (e.g., 30, 75, 150, etc.)
CALLS_PER_MINUTE = 5

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
    # 1) Streamlit secrets
    try:
        k = st.secrets.get("AlphaVantage_API_KEY", "")
        if k:
            return str(k).strip()
    except Exception:
        pass

    # 2) Env var fallback
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
# Alpha Vantage rate limiting (global)
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
# Alpha Vantage fetch
# Function: TIME_SERIES_DAILY_ADJUSTED
# Uses "5. adjusted close"
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)  # cache 12 hours
def fetch_daily_adj_closes_av(ticker: str) -> pd.Series:
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

    # Common AV throttling / errors
    if "Note" in js:
        # Rate-limit note
        raise RuntimeError(js["Note"])
    if "Error Message" in js:
        raise RuntimeError(js["Error Message"])

    ts_key = "Time Series (Daily)"
    if ts_key not in js:
        # Sometimes returns different keys or an unexpected payload
        raise RuntimeError(f"Unexpected Alpha Vantage response for {ticker}: {list(js.keys())[:5]}")

    rows = js[ts_key]  # dict of date -> o/h/l/c/adj/vol/etc
    if not rows:
        return pd.Series(dtype="float64", name=ticker)

    # Build series
    # Date strings are YYYY-MM-DD
    idx = []
    vals = []
    for d, rec in rows.items():
        # adjusted close: "5. adjusted close"
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


def build_price_matrix(tickers: list[str]) -> tuple[pd.DataFrame, int, list[str]]:
    """
    Sequential fetch (recommended for Alpha Vantage free tier).
    Returns: (DataFrame closes, failures_count, failed_tickers)
    """
    tickers = [normalize_ticker(t) for t in tickers]
    tickers = list(dict.fromkeys([t for t in tickers if t]))

    out = {}
    failed = []

    prog = st.progress(0)
    status = st.empty()

    n = len(tickers)
    for i, t in enumerate(tickers, start=1):
        status.write(f"Downloading {t} ({i}/{n}) …")
        try:
            ser = fetch_daily_adj_closes_av(t)
            if ser is not None and not ser.empty:
                out[t] = ser
            else:
                failed.append(t)
        except Exception:
            failed.append(t)

        prog.progress(i / n)

    status.empty()

    if not out:
        raise RuntimeError("No price data returned for any tickers (check API key / rate limits).")

    df = pd.DataFrame(out).sort_index()
    df = df.ffill()
    return df, len(failed), failed


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
# UI
# ============================================================
st.title("Relative Strength Scanner")
st.caption(f"As of: {utc_now_label()} • Benchmark: {BENCHMARK} • Data: Alpha Vantage (Adjusted Close)")

with st.sidebar:
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

    st.caption(f"Rate limit: ~{CALLS_PER_MINUTE} calls/min (adjust in code if your AV plan allows more)")

    run = st.button("Run Scan", use_container_width=True)


if not API_KEY:
    st.error(
        "Missing API key.\n\n"
        "Set **AlphaVantage_API_KEY** in Streamlit Secrets (recommended) "
        "or set environment variable **ALPHAVANTAGE_API_KEY**."
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

# Ensure benchmark included
bench_norm = normalize_ticker(BENCHMARK)
if bench_norm not in universe:
    universe = universe + [bench_norm]

st.write(f"Universe size: **{len(universe):,}** tickers (including benchmark).")


if run:
    st.info("Pulling price data… (Alpha Vantage rate-limited; cached 12 hours per ticker)")

    try:
        price_df, failures, failed_list = build_price_matrix(universe)
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

    if bench_norm not in price_df.columns:
        st.error(
            f"Benchmark {BENCHMARK} returned no data.\n\n"
            "This usually means:\n"
            "- Alpha Vantage rate limit hit\n"
            "- Symbol mismatch\n"
            "- Temporary API issue\n"
        )
        st.stop()

    # Keep only last ~300 trading days if you want speed in RS calc
    # (Optional) Uncomment:
    # price_df = price_df.tail(320)

    bench = price_df[bench_norm]

    # Compute raw RS ratios
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

    # Sort best → worst by primary timeframe, then RS 1Y as tie-breaker
    df_f = df_f.sort_values([primary_tf, "RS 1Y"], ascending=[False, False])

    st.success(
        f"Scan complete • {len(df_f):,} matches • "
        f"failed tickers during fetch: {failures}"
    )

    if failures and failed_list:
        with st.expander("Failed tickers (click to view)"):
            st.write(", ".join(failed_list[:500]))
            if len(failed_list) > 500:
                st.write(f"...and {len(failed_list) - 500} more")

    df_show = df_f.head(max_results).reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True, height=700)

    st.markdown(
        """
**How ranking works**  
- Each stock is compared to **SPY** over each timeframe (1W/1M/3M/6M/1Y).  
- Those relative-performance values are percentile-ranked **across your entire CSV universe** into **RS 1–99**.  
- RS 99 means “top ~1% of your list” for that timeframe.
"""
    )

    with st.expander("Troubleshooting"):
        st.write(
            "- If you see a message starting with **'Thank you for using Alpha Vantage…'**, you hit the rate limit.\n"
            "- On the free plan, keep CALLS_PER_MINUTE at 5 and expect larger universes to take time.\n"
            "- If you have a higher-tier Alpha Vantage plan, increase CALLS_PER_MINUTE in the code.\n"
            "- If some symbols fail, verify they are valid Alpha Vantage symbols (class shares often use dot format like BRK.B)."
        )
