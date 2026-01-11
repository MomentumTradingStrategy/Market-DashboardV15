import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Lookback: need enough trading days for 1Y (252). 420 calendar days is usually safe.
LOOKBACK_CAL_DAYS = 420

# RS horizons (trading days)
HORIZONS = {
    "RS 1W": 5,
    "RS 1M": 21,
    "RS 3M": 63,
    "RS 6M": 126,
    "RS 1Y": 252,
}

# Get API key securely (preferred order)
# 1) Streamlit secrets: st.secrets["POLYGON_API_KEY"]
# 2) Environment variable: POLYGON_API_KEY
def get_api_key() -> str:
    try:
        k = st.secrets.get("POLYGON_API_KEY", "")
        if k:
            return k.strip()
    except Exception:
        pass
    return os.environ.get("POLYGON_API_KEY", "").strip()


API_KEY = get_api_key()


def utc_now_label():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================
# Ticker normalization for Polygon
# - Polygon commonly uses dot classes (BRK.B) vs dash (BRK-B)
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
# Polygon / Massive AGGS fetch
# Endpoint:
# https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_daily_closes_polygon(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Returns a Series indexed by date (UTC midnight) with closes.
    Cached for 1 hour.
    """
    if not API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY.")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    s = requests.Session()
    r = s.get(url, params=params, timeout=25)
    r.raise_for_status()
    js = r.json()

    results = js.get("results", [])
    if not results:
        return pd.Series(dtype="float64", name=ticker)

    df = pd.DataFrame(results)
    # t is ms epoch
    dt = pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize()
    close = pd.Series(df["c"].values, index=dt, name=ticker).astype(float)
    close = close[~close.index.duplicated(keep="last")]
    return close


def build_price_matrix(tickers: list[str], start_date: str, end_date: str, max_workers: int) -> pd.DataFrame:
    """
    Concurrent fetch -> DataFrame of closes.
    """
    tickers = [normalize_ticker(t) for t in tickers]
    tickers = list(dict.fromkeys([t for t in tickers if t]))

    # Use one session per thread inside cached function call; okay.
    out = {}
    failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(fetch_daily_closes_polygon, t, start_date, end_date): t
            for t in tickers
        }

        for fut in as_completed(futs):
            t = futs[fut]
            try:
                ser = fut.result()
                if ser is not None and not ser.empty:
                    out[t] = ser
            except Exception:
                failures += 1

    if not out:
        raise RuntimeError("No price data returned for any tickers.")

    df = pd.DataFrame(out).sort_index()
    # forward-fill gaps
    df = df.ffill()
    return df, failures


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
st.caption(f"As of: {utc_now_label()} • Benchmark: {BENCHMARK}")

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

    # Concurrency (bigger = faster, but free plans rate-limit; keep sane defaults)
    max_workers = st.slider("Speed (parallel downloads)", 5, 40, 18, step=1)

    run = st.button("Run Scan", use_container_width=True)

# Guard: API key required
if not API_KEY:
    st.error(
        "Missing API key.\n\n"
        "Set POLYGON_API_KEY in Streamlit Secrets or as an environment variable. "
        "Users will not be prompted for it when you do that."
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

# Ensure benchmark in pull list
bench_norm = normalize_ticker(BENCHMARK)
if bench_norm not in universe:
    universe = universe + [bench_norm]

st.write(f"Universe size: **{len(universe):,}** tickers (including benchmark).")

if run:
    # Date range for Polygon
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=LOOKBACK_CAL_DAYS)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    st.info("Pulling price data… (cached for 1 hour)")
    prog = st.progress(0)

    # Fetch data
    price_df, failures = build_price_matrix(universe, start_str, end_str, max_workers=max_workers)
    prog.progress(1.0)

    if bench_norm not in price_df.columns:
        st.error(
            f"Benchmark {BENCHMARK} returned no data.\n\n"
            "This usually means either:\n"
            "- rate limiting on your plan\n"
            "- temporary API issue\n"
            "- benchmark symbol mismatch\n"
        )
        st.stop()

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

    # Convert each RS column to 1–99 percentile across entire universe
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
        f"Scan complete • {len(df_f):,} matches "
        f"(failed tickers during fetch: {failures})"
    )

    # Limit displayed results (still ranked against full universe)
    df_show = df_f.head(max_results).reset_index(drop=True)

    st.dataframe(df_show, use_container_width=True, height=700)

    st.markdown(
        """
**How ranking works**  
- Each stock is compared to **SPY** over each timeframe (1W/1M/3M/6M/1Y).  
- Those relative-performance values are percentile-ranked **across your entire CSV universe** into **RS 1–99**.  
- So RS 99 means “top ~1% of your list” for that timeframe.
"""
    )

    with st.expander("Troubleshooting"):
        st.write(
            "- If scans are slow or you see missing data on the free plan, reduce **Speed (parallel downloads)**.\n"
            "- Free tiers often rate-limit bulk calls; paid tiers are dramatically faster and more consistent.\n"
            "- If class-share tickers fail (BRK-B), try ensuring the CSV uses BRK-B or BRK.B; the app normalizes these."
        )


