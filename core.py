"""core.py — Pure functions returning plain dicts. No classes, no ORM.
Every function can later be exposed as an LLM tool without change.
"""
import os, duckdb, numpy as np, pandas as pd
from datetime import datetime
from typing import Optional
from indicators import *

_HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(_HERE, "database", "quantscanner.duckdb")
NIFTY_TICKER = "^NSEI"


def _conn():
    return duckdb.connect(DB)


def _closes_hlv(ticker: str, min_bars: int = 200, con=None):
    """Load OHLCV arrays for a ticker, sorted by date. Returns (closes, highs, lows, volumes, dates) or (None,)*5.
    Pass an existing con to avoid creating a new connection."""
    close_con = False
    if con is None:
        con = _conn()
        close_con = True
    rows = con.execute(
        "SELECT Date, Close, High, Low, Volume FROM DailyBars WHERE Ticker = ? ORDER BY Date",
        [ticker]
    ).fetchall()
    if close_con:
        con.close()
    if len(rows) < min_bars:
        return None, None, None, None, None
    dates = np.array([r[0] for r in rows])
    closes = np.array([float(r[1]) for r in rows], dtype=float)
    highs = np.array([float(r[2]) for r in rows], dtype=float)
    lows = np.array([float(r[3]) for r in rows], dtype=float)
    vols = np.array([float(r[4]) for r in rows], dtype=float)
    return closes, highs, lows, vols, dates


def _weekly_closes(ticker: str, min_bars: int = 50, con=None):
    close_con = False
    if con is None:
        con = _conn()
        close_con = True
    rows = con.execute(
        "SELECT Close FROM WeeklyBars WHERE Ticker = ? ORDER BY Date", [ticker]
    ).fetchall()
    if close_con:
        con.close()
    if len(rows) < min_bars:
        return None
    return np.array([float(r[0]) for r in rows], dtype=float)


def _batch_load_all(min_bars: int = 200, lookback: int = 250) -> dict:
    """Load OHLCV data for ALL tickers in one query.
    Returns dict: ticker -> (closes, highs, lows, vols, dates) as numpy arrays.
    Tickers with fewer than min_bars are excluded.
    """
    con = _conn()
    rows = con.execute("""
        SELECT Ticker, Date, Close, High, Low, Volume FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Ticker ORDER BY Date DESC) as rn
            FROM DailyBars
        ) sub WHERE rn <= ?
        ORDER BY Ticker, Date
    """, [lookback]).fetchall()
    con.close()

    # Organize rows by ticker
    raw: dict[str, list] = {}
    for row in rows:
        t = row[0]
        if t not in raw:
            raw[t] = {'dates': [], 'closes': [], 'highs': [], 'lows': [], 'vols': []}
        raw[t]['dates'].append(row[1])
        raw[t]['closes'].append(row[2])
        raw[t]['highs'].append(row[3])
        raw[t]['lows'].append(row[4])
        raw[t]['vols'].append(row[5])

    # Convert to numpy arrays, filter by min_bars
    result = {}
    for t, d in raw.items():
        n = len(d['closes'])
        if n < min_bars:
            continue
        result[t] = (
            np.array(d['closes'], dtype=float),
            np.array(d['highs'], dtype=float),
            np.array(d['lows'], dtype=float),
            np.array(d['vols'], dtype=float),
            np.array(d['dates']),
        )
    return result


def _load_nifty(con=None):
    """Load Nifty index data using existing con or new connection."""
    close_con = con is None
    if con is None:
        con = _conn()
    closes, _, _, _, _ = _closes_hlv(NIFTY_TICKER, 200, con=con)
    if close_con:
        con.close()
    return closes


def get_market_regime() -> dict:
    """Check Nifty 50 index vs 200 EMA for bull/bear regime."""
    closes, _, _, _, _ = _closes_hlv(NIFTY_TICKER, 200)
    if closes is None:
        return {"market_regime": "UNKNOWN", "index_close": 0, "index_ema200": 0}
    idx_close = safe_round(closes[-1])
    idx_ema200 = safe_round(ema_last(closes, 200))
    regime = "BULLISH" if closes[-1] >= idx_ema200 else "BEARISH"
    return {"market_regime": regime, "index_close": idx_close, "index_ema200": idx_ema200}


def get_strategies() -> list:
    """Return list of available strategy names (matches Angular frontend + C# backtest)."""
    return [
        "All",
        "JustNifty Positional",
        "JustNifty HCT",
        "JustNifty LRHR",
        "Quant HCT Pullback",
        "Quant LRHR Base",
        "MOMCON",
        "VAL",
        "VBO",
        "MOMACC",
        "CBO",
        "DPA",
        "RSML",
    ]


def get_stock_scan(strategy: str = "All") -> dict:
    """Run the full scan. Returns dict with market_regime, results list.
    Each result: {ticker, sector, price, score, strategy, conviction, indicators...}
    Uses batch DB loading for speed: ~1 query instead of ~1700.
    """
    regime = get_market_regime()

    # Batch-load all ticker data in ONE query
    all_data = _batch_load_all(min_bars=200, lookback=250)

    # Load Nifty data
    n_closes = _load_nifty()
    if n_closes is None:
        return {**regime, "results": [], "total_scored": 0}

    # Sector mapping
    con = _conn()
    stocks = con.execute("SELECT Ticker, Sector FROM StockMetadatas").fetchall()
    sectors = {r[0]: r[1] or "NSE" for r in stocks}
    con.close()

    # Pre-compute RS percentile ranks (mirrors C# RsRank — 3M return percentile across all stocks)
    _rs_returns = {t: calc_return(d[0], 60) for t, d in all_data.items() if len(d[0]) >= 60}
    _rs_rank_map = {}
    if _rs_returns:
        _sorted_uniq = sorted(set(_rs_returns.values()))
        _rank_lookup = {v: i / len(_sorted_uniq) * 100 for i, v in enumerate(_sorted_uniq)}
        _rs_rank_map = {t: _rank_lookup.get(ret, 50.0) for t, ret in _rs_returns.items()}

    idx_3m = calc_return(n_closes, 60)
    idx_6m = calc_return(n_closes, 120)

    results = []
    for ticker, sector in sectors.items():
        ticker_data = all_data.get(ticker)
        if ticker_data is None:
            continue
        try:
            closes, highs, lows, vols, dates = ticker_data
            price = closes[-1]
            ema8, ema10, ema21, ema50, ema200 = multi_ema_last(closes)
            jnsar = jnsar_last(closes, highs, lows)
            atr_full = atr(highs, lows, closes)
            atr_last = atr_full[-1] if len(atr_full) > 0 else 0
            rsi_val = rsi_last(closes)
            adx_val = adx_last(highs, lows, closes)
            macd_line, macd_signal = macd(closes)
            macd_bull = macd_line[-1] > macd_signal[-1]

            vol_pct_rank = vol_percentile_rank(atr_full)
            is_atr_coiled = vol_pct_rank < 30
            is_squeeze = vol_pct_rank < 20 and rsi_val < 60

            s3m = calc_return(closes, 60)
            s6m = calc_return(closes, 120)

            max52 = max_52_high(highs)
            disc52w = (max52 - price) / max52 if max52 > 0 else 0

            fib618, swh, swl = swing_fib618(closes, highs, lows)
            zsc = z_score_last(closes)
            vs = volume_score(closes, vols)
            poc = point_of_control(closes, vols)
            ytd_v = ytd_vwap(closes, dates, vols)

            obv_arr = obv(closes, vols)
            cmf_value = cmf_last(highs, lows, closes, vols)
            # Need a small cmf slice for cmf[-5] inflection check
            cmf_slice = cmf(highs, lows, closes, vols) if len(closes) >= 30 else np.array([0.0])
            cmf_inflection = cmf_slice[-5] <= 0 if len(cmf_slice) >= 5 else True
            obv_up = obv_arr[-1] > obv_arr[-20] if len(obv_arr) >= 20 else False
            obv_up_10 = obv_arr[-1] > obv_arr[-10] if len(obv_arr) >= 10 else False

            chand = chandelier_exit(highs, lows, closes)
            chand_last = float(chand[-1]) if len(chand) > 0 else 0

            rs_rank = _rs_rank_map.get(ticker, 50.0)

            trend_score = 0
            if price > ema50:
                trend_score += 5
            if ema50 > ema200:
                trend_score += 5
            if adx_val > 25:
                trend_score += 5
            trend_score += 5

            rs_score = 0
            if s3m > idx_3m:
                rs_score += 10
            if s6m > idx_6m:
                rs_score += 10

            proximity_score = 10 if disc52w <= 0.05 else (6 if disc52w <= 0.10 else 0)
            vol_acc_score = vs
            vol_setup_score = (5 if is_atr_coiled else 0) + (5 if is_squeeze else 0)
            momentum_score = 10 if 50 <= rsi_val <= 70 else (5 if 40 < rsi_val < 50 else 0)

            inst_score = 0
            lookback = min(20, len(closes) - 1)
            up_vol = sum(vols[-lookback:][closes[-lookback:] > closes[-lookback - 1:-1]]) if lookback >= 1 else 0
            dn_vol = sum(vols[-lookback:][closes[-lookback:] < closes[-lookback - 1:-1]]) if lookback >= 1 else 0
            up_count = int((closes[-lookback:] > closes[-lookback - 1:-1]).sum()) if lookback >= 1 else 0
            dn_count = lookback - up_count if lookback >= 1 else 0
            avg_up = up_vol / up_count if up_count > 0 else 0
            avg_dn = dn_vol / dn_count if dn_count > 0 else 0
            if avg_up > avg_dn * 1.5:
                inst_score = 10
            elif avg_up > avg_dn:
                inst_score = 5

            total_score = trend_score + rs_score + vol_acc_score + vol_setup_score + momentum_score + inst_score

            matched = []
            if price > ema200 and ema8 > ema21 and price > ema10 and macd_bull:
                matched.append("JustNifty Positional")
            if price > ema200 and price > ema21:
                matched.append("JustNifty HCT")
            if price > ema200 and price < ema200 * 1.05:
                matched.append("JustNifty LRHR")
            # DPA: dip accumulation with institutional support
            if zsc < -0.5 and cmf_value > 0 and obv_up and price > ema200:
                matched.append("DPA")
            # RSML: relative strength momentum leader
            if rs_rank > 70 and price > ema21 and ema8 > ema21:
                matched.append("RSML")
            # Quant strategies
            ytd_ok = (price >= ytd_v * 0.97 and price <= ytd_v * 1.05) or (price >= poc * 0.97 and price <= poc * 1.05)
            if ytd_ok and obv_up and cmf_value > 0 and vol_pct_rank < 30 and zsc >= -1.0 and zsc <= 0.5 and price > chand_last:
                matched.append("Quant HCT Pullback")
            if zsc < -1.0 and disc52w >= 0.15 and cmf_value > 0 and cmf_inflection and obv_up_10 and vol_pct_rank < 50 and price > poc:
                matched.append("Quant LRHR Base")
            # Momentum / value
            if total_score >= 25 and macd_bull and price > ema50:
                matched.append("MOMCON")
            if zsc < 0 and cmf_value > 0 and obv_up:
                matched.append("VAL")
            if is_squeeze and vol_pct_rank < 20 and vs >= 5:
                matched.append("VBO")
            if price > ema21 and obv_up and rsi_val > 50:
                matched.append("MOMACC")
            # CBO: Bollinger inside Keltner = volatility compression
            bb_up = bollinger_last(closes)[0]
            bb_low = bollinger_last(closes)[2]
            kc_up = keltner_last(highs, lows, closes)[0]
            kc_low = keltner_last(highs, lows, closes)[2]
            if bb_up > 0 and bb_up < kc_up and bb_low > kc_low and vol_pct_rank < 40:
                matched.append("CBO")

            strategy_match = matched[0] if matched else "None"
            conviction = "HIGH" if total_score >= 50 else ("MEDIUM" if total_score >= 25 else "LOW")

            target1, target2, stop = volatility_fib_targets(closes)
            if price > jnsar and stop < jnsar:
                stop = jnsar
            if price > chand_last and chand_last > stop:
                stop = chand_last

            results.append({
                "ticker": ticker.replace(".NS", ""),
                "sector": sector,
                "price": safe_round(price),
                "score": total_score,
                "strategy": strategy_match,
                "conviction": conviction,
                "ema8": safe_round(ema8), "ema10": safe_round(ema10),
                "ema21": safe_round(ema21), "ema50": safe_round(ema50), "ema200": safe_round(ema200),
                "jnsar": safe_round(jnsar), "fib618": safe_round(fib618),
                "atr14": safe_round(atr_last), "rsi14": safe_round(rsi_val),
                "adx14": safe_round(adx_val), "z_score": safe_round(zsc),
                "discount_52w": safe_round(disc52w * 100),
                "volume_score": vol_acc_score,
                "poc": safe_round(poc), "ytd_vwap": safe_round(ytd_v),
                "chandelier": safe_round(chand_last),
                "obv": safe_round(float(obv_arr[-1]), 0),
                "cmf": safe_round(cmf_value),
                "vol_pct_rank": safe_round(vol_pct_rank, 1),
                "rs_sharpe": safe_round(rolling_sharpe(closes)),
                "stop_loss": safe_round(stop),
                "target1": safe_round(target1),
                "target2": safe_round(target2),
                "trend_score": trend_score, "rs_score": rs_score,
                "proximity_score": proximity_score, "vol_acc_score": vol_acc_score,
                "vol_setup_score": vol_setup_score, "momentum_score": momentum_score,
                "inst_score": inst_score,
            })
        except Exception:
            continue

    if strategy != "All":
        results = [r for r in results if r["strategy"] == strategy]
        # Specific strategy: show all matches, no score filter
        results.sort(key=lambda r: r["score"], reverse=True)
        return {**regime, "results": results, "total_scored": len(results)}

    # "All" view: score thresholds (JustNifty >= 65, others >= 60)
    results = [r for r in results if r["score"] >= (65 if r["strategy"].startswith("JustNifty") else 60)]
    results.sort(key=lambda r: r["score"], reverse=True)
    return {**regime, "results": results, "total_scored": len(results)}


def get_stock_chart(ticker: str, limit: int = 250) -> dict:
    """Candlestick data with indicators for a single ticker."""
    t = ticker + ".NS" if not ticker.endswith(".NS") else ticker
    try:
        closes, highs, lows, vols, dates = _closes_hlv(t, 50)
        if closes is None:
            return {"ticker": ticker, "candles": [], "error": "Insufficient data"}
    except Exception as e:
        return {"ticker": ticker, "candles": [], "error": str(e)}

    slices = lambda arr: arr[-limit:] if len(arr) > limit else arr
    cs, hs, ls, vs, ds = slices(closes), slices(highs), slices(lows), slices(vols), slices(dates)
    ema8_arr = ema(cs, 8)
    ema21_arr = ema(cs, 21)
    ema200_val = ema_last(cs, 200)
    jnsar_arr = jnsar(cs, hs, ls)
    fib618, _, _ = swing_fib618(cs, hs, ls)
    macd_l, macd_s = macd(cs)

    candles = []

    # Fetch opens separately (not in _closes_hlv)
    con = _conn()
    rows = con.execute(
        "SELECT Date, Open FROM DailyBars WHERE Ticker = ? ORDER BY Date", [t]
    ).fetchall()
    con.close()
    opens_map = {r[0]: float(r[1]) for r in rows}

    for i in range(len(ds)):
        d = ds[i]
        candles.append({
            "date": d.isoformat() if hasattr(d, 'isoformat') else str(d),
            "open": safe_round(opens_map.get(d, cs[i])),
            "high": safe_round(hs[i]), "low": safe_round(ls[i]),
            "close": safe_round(cs[i]), "volume": int(vs[i]),
            "ema8": safe_round(ema8_arr[i]),
            "ema21": safe_round(ema21_arr[i]),
            "ema200": safe_round(ema200_val) if i == len(ds) - 1 else None,
            "jnsar": safe_round(jnsar_arr[i]),
            "fib618": safe_round(fib618) if i == len(ds) - 1 else None,
            "macd_line": safe_round(macd_l[i]),
            "macd_signal": safe_round(macd_s[i]),
            "macd_histogram": safe_round(macd_l[i] - macd_s[i]),
        })
    return {"ticker": ticker, "candles": candles}


def get_sector_rotation() -> dict:
    """RRG quadrant analysis for sector indices with historical trails."""
    con = _conn()
    sectors = con.execute("SELECT DISTINCT Ticker FROM SectorDailyBars WHERE Ticker != ? ORDER BY Ticker", [NIFTY_TICKER]).fetchall()
    nifty = con.execute("SELECT Date, Close FROM SectorDailyBars WHERE Ticker = ? ORDER BY Date", [NIFTY_TICKER]).fetchall()
    con.close()
    if not nifty or not sectors:
        return {"sectors": [], "leading": [], "improving": [], "weakening": [], "lagging": [], "rotation_signal": False}

    n_close = np.array([float(r[1]) for r in nifty])
    n_date = [r[0] for r in nifty]
    s_names = {"^NSEBANK": "Bank", "^CNXAUTO": "Auto", "^CNXIT": "IT",
               "^CNXPHARMA": "Pharma", "^CNXMETAL": "Metal", "^CNXENERGY": "Energy",
               "^CNXFMCG": "FMCG", "^CNXMEDIA": "Media", "^CNXREALTY": "Realty",
               "^CNXPSUBANK": "PSU Bank", "^CNXINFRA": "Infrastructure",
               "NIFTY_FIN_SERVICE.NS": "Financial Services",
               "NIFTY_OIL_AND_GAS.NS": "Oil & Gas",
               "^CNXCONSUM": "Consumer Durables"}

    def _ema(arr, p):
        out = np.empty_like(arr)
        k = 2.0 / (p + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    def _rrg_z(rs_arr):
        """Compute current RRG z-scores from RS array."""
        rs_st = _ema(rs_arr, 10)
        rs_lt = _ema(rs_arr, 40)
        mom = rs_st - rs_lt
        lb = min(250, len(rs_st))
        st_s = rs_st[-lb:]
        rz = (rs_st[-1] - st_s.mean()) / st_s.std(ddof=0) if st_s.std(ddof=0) > 0 else 0
        mo_s = mom[-lb:]
        mz = (mom[-1] - mo_s.mean()) / mo_s.std(ddof=0) if mo_s.std(ddof=0) > 0 else 0
        return rz, mz

    results = []
    for (ticker,) in sectors:
        con = _conn()
        rows = con.execute("SELECT Date, Close FROM SectorDailyBars WHERE Ticker = ? ORDER BY Date", [ticker]).fetchall()
        con.close()
        if len(rows) < 250:
            continue
        c = np.array([float(r[1]) for r in rows])
        min_l = min(len(c), len(n_close))
        rs = c[-min_l:] / n_close[-min_l:]

        # Current RRG position
        rs_z, mo_z = _rrg_z(rs)

        # Historical trail: compute RRG at ~20 weekly points going back
        trail = []
        for offset in range(5, min(100, len(rs)), 5):
            rs_slice = rs[:-offset] if offset < len(rs) else rs
            if len(rs_slice) >= 250:
                hrz, hmz = _rrg_z(rs_slice)
                trail.append({"rs_ratio": safe_round(hrz), "rs_momentum": safe_round(hmz)})
        trail.append({"rs_ratio": safe_round(rs_z), "rs_momentum": safe_round(mo_z)})  # current

        quad = "Leading" if rs_z > 0 and mo_z > 0 else ("Weakening" if rs_z > 0 else ("Improving" if mo_z > 0 else "Lagging"))

        results.append({
            "ticker": ticker,
            "name": s_names.get(ticker, ticker.replace("^", "")),
            "rs_ratio": safe_round(rs_z), "rs_momentum": safe_round(mo_z),
            "quadrant": quad, "price": safe_round(c[-1]),
            "trail": trail,
        })

    leading = [r for r in results if r["quadrant"] == "Leading"]
    improving = [r for r in results if r["quadrant"] == "Improving"]
    weakening = [r for r in results if r["quadrant"] == "Weakening"]
    lagging = [r for r in results if r["quadrant"] == "Lagging"]
    return {
        "sectors": results, "leading": leading, "improving": improving,
        "weakening": weakening, "lagging": lagging,
        "rotation_signal": len(improving) >= 2 and len(weakening) >= 2,
    }


def run_rotation_backtest(params: dict) -> dict:
    """Rotation backtest: buy sectors transitioning Lagging→Improving, sell on Leading→Weakening.
    Uses weekly rebalance (every 5 trading days) over SectorDailyBars.
    Returns equity_curve, trades, summary stats.
    """
    capital = float(params.get("starting_capital", 100000))

    con = _conn()
    all_rows = con.execute("SELECT Ticker, Date, Open, Close FROM SectorDailyBars ORDER BY Ticker, Date").fetchall()
    con.close()
    if not all_rows:
        return {"error": "No sector data"}

    grouped = {}
    for r in all_rows:
        grouped.setdefault(r[0], []).append(r)
    grouped = {k: sorted(v, key=lambda x: x[1]) for k, v in grouped.items()}

    # Pre-convert all ticker data to numpy arrays for fast lookups
    ticker_data = {}
    for ticker, bars in grouped.items():
        ticker_data[ticker] = {
            "dates": np.array([r[1] for r in bars]),
            "opens": np.array([float(r[2]) for r in bars]),
            "closes": np.array([float(r[3]) for r in bars]),
        }

    if "^NSEI" not in grouped:
        return {"error": "No Nifty index data"}
    nifty = grouped["^NSEI"]
    n_closes = np.array([float(r[3]) for r in nifty])  # Close
    n_dates = [r[1] for r in nifty]

    sector_names = {"^NSEBANK": "Bank", "^CNXAUTO": "Auto", "^CNXIT": "IT",
                    "^CNXPHARMA": "Pharma", "^CNXMETAL": "Metal", "^CNXENERGY": "Energy",
                    "^CNXFMCG": "FMCG", "^CNXMEDIA": "Media", "^CNXREALTY": "Realty",
                    "^CNXPSUBANK": "PSU Bank", "^CNXINFRA": "Infrastructure",
                    "NIFTY_FIN_SERVICE.NS": "Financial Services",
                    "NIFTY_OIL_AND_GAS.NS": "Oil & Gas",
                    "^CNXCONSUM": "Consumer Durables"}

    def _ema(arr, p):
        out = np.empty_like(arr)
        k = 2.0 / (p + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    def _compute_rrg(ticker_bars, nifty_slice, n_end, lookback=250):
        """Compute RRG quadrant at a specific point."""
        closes = np.array([float(r[2]) for r in ticker_bars])
        min_l = min(len(closes), len(nifty_slice))
        if min_l < lookback:
            return None
        rs = closes[-min_l:] / nifty_slice[-min_l:]
        rs_st = _ema(rs, 10)
        rs_lt = _ema(rs, 40)
        mom = rs_st - rs_lt
        lb = min(lookback, len(rs_st))
        st_s = rs_st[-lb:]
        rs_z = (rs_st[-1] - st_s.mean()) / st_s.std(ddof=0) if st_s.std(ddof=0) > 0 else 0
        mo_s = mom[-lb:]
        mo_z = (mom[-1] - mo_s.mean()) / mo_s.std(ddof=0) if mo_s.std(ddof=0) > 0 else 0
        if rs_z > 0 and mo_z > 0:
            q = "Leading"
        elif rs_z > 0:
            q = "Weakening"
        elif mo_z > 0:
            q = "Improving"
        else:
            q = "Lagging"
        return {"quadrant": q, "rs_z": rs_z, "mo_z": mo_z}

    # Walk weekly through time
    start_idx = 50  # need 50 bars for 1M momentum
    step = 5  # weekly
    balance = capital
    peak = capital
    trades = []
    equity = [{"date": n_dates[start_idx].isoformat(), "balance": capital, "drawdown_pct": 0}]
    positions = {}  # ticker -> {entry_idx, entry_price}

    for idx in range(start_idx + step, len(n_dates), step):
        current_date = n_dates[idx]
        prev_date = n_dates[idx - step]  # last week's date (signal reference)

        # Compute 1-month momentum using LAST WEEK's close + numpy fast lookups
        prev_quads = {}
        for ticker, td in ticker_data.items():
            if ticker == "^NSEI":
                continue
            if len(td["dates"]) <= idx:
                continue
            i_prev = np.searchsorted(td["dates"], prev_date, side="right") - 1
            if i_prev < 25:
                continue
            mom_1m = (td["closes"][i_prev] - td["closes"][i_prev - 21]) / td["closes"][i_prev - 21] * 100
            i_curr = np.searchsorted(td["dates"], current_date, side="right") - 1
            curr_open = float(td["opens"][i_curr]) if i_curr >= 0 else 0
            curr_close = float(td["closes"][i_curr]) if i_curr >= 0 else 0
            prev_quads[ticker] = {"curr_price": curr_open,  # execute at open
                                   "curr_close": curr_close,  # for peak tracking
                                   "mo_z": mom_1m}

        # Top-3 momentum rotation: rank sectors by RS-Momentum, hold top 3
        ranked = sorted([(pq["mo_z"], ticker, pq["curr_price"])
                         for ticker, pq in prev_quads.items()], key=lambda x: -x[0])
        top_n = 2
        top = ranked[:top_n]

        # Exit: sell if not the #1 momentum sector, rotate to new #1
        best_ticker = ranked[0][1] if ranked else None
        for ticker in list(positions.keys()):
            if ticker != best_ticker:
                pq = prev_quads.get(ticker)
                if pq is None:
                    continue
                exit_price = pq["curr_price"]
                p = positions[ticker]
                gross_ret = (exit_price - p["entry_price"]) / p["entry_price"] * 100
                trades.append({
                    "ticker": ticker, "signal": "SELL",
                    "name": sector_names.get(ticker, ticker),
                    "entry_date": n_dates[p["entry_idx"]].isoformat(),
                    "exit_date": current_date.isoformat(),
                    "entry_price": safe_round(p["entry_price"]),
                    "exit_price": safe_round(exit_price),
                    "return_pct": safe_round(gross_ret),
                    "days_held": (n_dates[idx] - n_dates[p["entry_idx"]]).days,
                })
                balance += p["shares"] * exit_price
                del positions[ticker]

        # Buy top N sectors equally (only if not already held)
        to_buy = [(t, p) for _, t, p in top if t not in positions]
        if to_buy:
            alloc = balance / len(to_buy)
            for ticker, entry_price in to_buy:
                shares = int(alloc / entry_price) if entry_price > 0 else 0
                if shares > 0:
                    cost = shares * entry_price
                    if cost <= balance:
                        balance -= cost
                        positions[ticker] = {"entry_idx": idx, "entry_price": entry_price, "shares": shares,
                                              "peak_price": entry_price}

        # Mark to market
        mtm = balance
        for t, p in positions.items():
            if t in prev_quads:
                mtm += p["shares"] * prev_quads[t]["curr_price"]
        if mtm > peak:
            peak = mtm
        dd = (peak - mtm) / peak * 100 if peak > 0 else 0
        equity.append({"date": current_date.isoformat(), "balance": safe_round(mtm), "drawdown_pct": safe_round(dd)})

    # Close remaining positions
    for ticker, p in positions.items():
        if ticker in prev_quads:
            exit_price = prev_quads[ticker]["curr_price"]
        else:
            exit_price = p["entry_price"]
        gross_ret = (exit_price - p["entry_price"]) / p["entry_price"] * 100
        trades.append({
            "ticker": ticker,
            "signal": "SELL",
            "name": sector_names.get(ticker, ticker),
            "entry_date": n_dates[p["entry_idx"]].isoformat(),
            "exit_date": n_dates[-1].isoformat(),
            "entry_price": safe_round(p["entry_price"]),
            "exit_price": safe_round(exit_price),
            "return_pct": safe_round(gross_ret),
            "days_held": (n_dates[-1] - n_dates[p["entry_idx"]]).days,
        })
        balance += p["shares"] * exit_price

    end_bal = balance
    total_return = (end_bal - capital) / capital * 100 if capital > 0 else 0
    nifty_ret = (n_closes[-1] - n_closes[start_idx]) / n_closes[start_idx] * 100

    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t["return_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["return_pct"] for t in losses) / len(losses) if losses else 0
    max_dd = max(e["drawdown_pct"] for e in equity)

    # Nifty equity curve
    nifty_eq = []
    for e in equity:
        n_idx = min(len(n_closes) - 1, start_idx + len(nifty_eq))
        nv = n_closes[n_idx]
        nifty_ret_pct = (nv - n_closes[start_idx]) / n_closes[start_idx] * 100
        nifty_eq.append(nifty_ret_pct)

    return {
        "starting_capital": capital,
        "ending_capital": safe_round(end_bal),
        "total_profit": safe_round(end_bal - capital),
        "return_pct": safe_round(total_return),
        "nifty_return": safe_round(nifty_ret),
        "max_drawdown_pct": safe_round(max_dd),
        "win_rate": safe_round(win_rate),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "avg_win": safe_round(avg_win),
        "avg_loss": safe_round(avg_loss),
        "trades": trades,
        "equity_curve": equity,
        "nifty_curve": nifty_eq,
    }


def run_backtest(params: dict) -> dict:
    """Portfolio simulation. params keys: strategy, starting_capital, max_positions,
    risk_per_trade_pct, sizing_model, transaction_cost_pct, slippage_pct.
    Returns equity_curve, trades, summary stats.
    """
    strategy = params.get("strategy", "JustNifty Positional")
    capital = float(params.get("starting_capital", 100000))
    max_pos = int(params.get("max_positions", 5))
    risk_pct = float(params.get("risk_per_trade_pct", 2.0)) / 100
    cost_pct = float(params.get("transaction_cost_pct", 0.1)) / 100
    slippage_pct = float(params.get("slippage_pct", 0.1)) / 100

    con = _conn()
    stocks = con.execute("SELECT Ticker FROM StockMetadatas").fetchall()
    tickers = [r[0] for r in stocks]
    nifty_rows = con.execute("SELECT Date, Close FROM DailyBars WHERE Ticker = ? ORDER BY Date", [NIFTY_TICKER]).fetchall()
    con.close()

    if not nifty_rows:
        return {"error": "No Nifty data"}

    all_dates = [r[0] for r in nifty_rows]
    n_prices = np.array([float(r[1]) for r in nifty_rows])

    # Batch-load all ticker data for full universe
    all_data = _batch_load_all(min_bars=200, lookback=600)
    ticker_data = {t: {"closes": d[0], "highs": d[1], "lows": d[2], "volumes": d[3], "dates": d[4]}
                   for t, d in all_data.items()}

    # Pre-compute indicator arrays ONCE per ticker (avoids recomputing at every step)
    for t, td in list(ticker_data.items()):
        c = td["closes"]
        try:
            td["ema8"] = ema(c, 8)
            td["ema10"] = ema(c, 10)
            td["ema21"] = ema(c, 21)
            td["ema50"] = ema(c, 50)
            td["ema200"] = ema(c, 200)
            td["macd_line"], td["macd_sig"] = macd(c)
            td["rsi"] = rsi(c)
            td["adx"] = adx(td["highs"], td["lows"], c)
            td["atr"] = atr(td["highs"], td["lows"], c)
            td["obv"] = obv(c, td["volumes"])
            td["cmf"] = cmf(td["highs"], td["lows"], c, td["volumes"])
            td["bb_u"], td["bb_m"], td["bb_l"] = bollinger(c)
            td["chandelier"] = chandelier_exit(td["highs"], td["lows"], c)
        except Exception:
            del ticker_data[t]
            continue

    balance = capital
    equity = [{"date": all_dates[0].isoformat(), "balance": capital, "drawdown_pct": 0}]
    trades = []
    pos = {}  # ticker -> {entry_date, entry_price, shares}
    peak = capital

    # Simulate monthly rebalance over full available data
    start_idx = 0
    step = 20  # ~monthly

    for idx in range(start_idx + step, len(all_dates), step):
        current_date = all_dates[idx]
        prev_idx = idx - 1  # signal from prev close, execute at current
        candidates = []
        for t, td in ticker_data.items():
            c = td["closes"]
            d_arr = td["dates"]
            if len(c) < 200:
                continue
            # Find bar index by date (ticker data has different date range than Nifty)
            i_c = int(np.searchsorted(d_arr, current_date, side="right")) - 2  # signal = prev close
            if i_c < 200:
                continue
            price = float(c[i_c + 1]) if i_c + 1 < len(c) else float(c[i_c])  # execution at current close
            if price <= 0:
                continue

            ema8, ema10, ema21, ema50, ema200 = (td["ema8"][i_c], td["ema10"][i_c], td["ema21"][i_c], td["ema50"][i_c], td["ema200"][i_c])

            if strategy == "JustNifty Positional":
                if price > ema200 and ema8 > ema21 and price > ema10 and td["macd_line"][i_c] > td["macd_sig"][i_c]:
                    candidates.append((t, price))
            elif strategy == "JustNifty HCT":
                if price > ema200 and price > ema21:
                    candidates.append((t, price))
            elif strategy == "JustNifty LRHR":
                if price > ema200 and price < ema200 * 1.05:
                    candidates.append((t, price))
            elif strategy == "Quant HCT Pullback":
                c_sig = c[:i_c+1]; v_sig = td["volumes"][:i_c+1]; h_sig = td["highs"][:i_c+1]; l_sig = td["lows"][:i_c+1]
                ytd_v = ytd_vwap(c_sig, td["dates"][:i_c+1], v_sig)
                poc_v = point_of_control(c_sig, v_sig)
                zsc = z_score_last(c_sig)
                chand_l = td["chandelier"][i_c]
                obv_up = td["obv"][i_c] > td["obv"][i_c-20] if i_c >= 20 else False
                cmf_v = td["cmf"][i_c]
                vpr = vol_percentile_rank(td["atr"][:i_c+1])
                ytd_ok = (price >= ytd_v * 0.97 and price <= ytd_v * 1.05) or (price >= poc_v * 0.97 and price <= poc_v * 1.05)
                if ytd_ok and obv_up and cmf_v > 0 and vpr < 30 and zsc >= -1.0 and zsc <= 0.5 and price > chand_l:
                    candidates.append((t, price))
            elif strategy == "Quant LRHR Base":
                c_sig = c[:i_c+1]; v_sig = td["volumes"][:i_c+1]; h_sig = td["highs"][:i_c+1]; l_sig = td["lows"][:i_c+1]
                zsc = z_score_last(c_sig)
                max52 = float(h_sig.max())
                disc52w = (max52 - price) / max52 if max52 > 0 else 0
                cmf_v = td["cmf"][i_c]
                poc_v = point_of_control(c_sig, v_sig)
                vpr = vol_percentile_rank(td["atr"][:i_c+1])
                obv_up_10 = td["obv"][i_c] > td["obv"][i_c-10] if i_c >= 10 else False
                cmf_inflection = td["cmf"][i_c-5] <= 0 if i_c >= 5 else True
                if zsc < -1.0 and disc52w >= 0.15 and cmf_v > 0 and cmf_inflection and obv_up_10 and vpr < 50 and price > poc_v:
                    candidates.append((t, price))
            elif strategy == "MOMCON":
                total = (5 if price>ema50 else 0)+(5 if ema50>ema200 else 0)+(5 if td["adx"][i_c]>25 else 0)+5
                if total >= 10 and td["macd_line"][i_c] > td["macd_sig"][i_c] and price > ema50:
                    candidates.append((t, price))
            elif strategy == "VAL":
                if z_score_last(c[:i_c+1]) < 0 and td["cmf"][i_c] > 0 and (td["obv"][i_c] > td["obv"][i_c-20] if i_c >= 20 else False):
                    candidates.append((t, price))
            elif strategy == "VBO":
                rsi_v = td["rsi"][i_c]
                vpr = vol_percentile_rank(td["atr"][:i_c+1])
                vs = volume_score(c[:i_c+1], td["volumes"][:i_c+1])
                if vpr < 20 and rsi_v < 60 and vs >= 5:
                    candidates.append((t, price))
            elif strategy == "MOMACC":
                rsi_v = td["rsi"][i_c]
                obv_up = td["obv"][i_c] > td["obv"][i_c-21] if i_c >= 21 else False
                if price > ema21 and obv_up and rsi_v > 50:
                    candidates.append((t, price))
            elif strategy == "CBO":
                vpr = vol_percentile_rank(td["atr"][:i_c+1])
                if td["bb_u"][i_c] > 0 and td["bb_u"][i_c] < keltner_last(td["highs"][:i_c+1], td["lows"][:i_c+1], c[:i_c+1])[0] and td["bb_l"][i_c] > keltner_last(td["highs"][:i_c+1], td["lows"][:i_c+1], c[:i_c+1])[2] and vpr < 40:
                    candidates.append((t, price))
            elif strategy == "DPA":
                if z_score_last(c[:i_c+1]) < -0.5 and td["cmf"][i_c] > 0 and (td["obv"][i_c] > td["obv"][i_c-20] if i_c >= 20 else False) and price > ema200:
                    candidates.append((t, price))
            elif strategy == "RSML":
                if price > ema21 and ema8 > ema21:
                    candidates.append((t, price))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:max_pos]

        # Quant desk exits: ATR trailing stop, hard stop, profit trim (before rebalance)
        atr_mult = 3.0
        for ticker in list(pos.keys()):
            td_e = ticker_data.get(ticker)
            if td_e is None:
                continue
            c_e = td_e["closes"]
            d_e = td_e["dates"]
            if len(c_e) <= 0:
                continue
            i_e = int(np.searchsorted(d_e, current_date, side="right")) - 1
            if i_e < 0:
                continue
            curr_price = float(c_e[i_e])
            p = pos[ticker]
            p["peak_price"] = max(p["peak_price"], curr_price)

            # Hard stop: exit if price drops 2x ATR below entry
            if curr_price < p["entry_price"] - 2.0 * p["entry_atr"]:
                exit_price = curr_price / (1 + slippage_pct)
                p["shares"] = 0

            # ATR trailing stop: trail 3x ATR from peak
            elif curr_price < p["peak_price"] - atr_mult * p["entry_atr"]:
                exit_price = curr_price / (1 + slippage_pct)
                p["shares"] = 0

            # Profit target: trim 50% at 2x ATR gain
            elif curr_price >= p["entry_price"] + 2.0 * p["entry_atr"] and not p["trimmed"]:
                trim = p["shares"] // 2
                if trim > 0:
                    exit_price = curr_price / (1 + slippage_pct)
                    p["shares"] -= trim
                    p["trimmed"] = True
                    gross_ret = (exit_price - p["entry_price"]) / p["entry_price"]
                    net_ret = gross_ret - 2 * cost_pct - slippage_pct
                    profit = trim * (exit_price - p["entry_price"])
                    trades.append({
                        "ticker": ticker.replace(".NS", ""),
                        "entry_date": p["entry_date"].isoformat(),
                        "entry_price": safe_round(p["entry_price"]),
                        "exit_date": current_date.isoformat(),
                        "exit_price": safe_round(exit_price),
                        "shares": trim, "profit": safe_round(profit),
                        "profit_pct": safe_round(net_ret * 100),
                        "exit_reason": "ProfitTarget",
                    })
                    balance += trim * exit_price

        # Close remaining positions (exit reason: rebalance or stop/trail)
        for ticker in list(pos.keys()):
            p = pos[ticker]
            if p["shares"] > 0 and ticker not in [c[0] for c in candidates]:
                td_r = ticker_data.get(ticker)
                if td_r:
                    i_r = int(np.searchsorted(td_r["dates"], current_date, side="right")) - 1
                    exit_price = float(td_r["closes"][i_r]) if i_r >= 0 else p["entry_price"]
                else:
                    exit_price = p["entry_price"]
                gross_ret = (exit_price - p["entry_price"]) / p["entry_price"]
                net_ret = gross_ret - 2 * cost_pct - slippage_pct
                profit = p["shares"] * (exit_price - p["entry_price"])
                exit_reason = "StopOut" if p["shares"] <= 0 else "Rebalance"
                trades.append({
                    "ticker": ticker.replace(".NS", ""),
                    "entry_date": p["entry_date"].isoformat(),
                    "entry_price": safe_round(p["entry_price"]),
                    "exit_date": current_date.isoformat(),
                    "exit_price": safe_round(exit_price),
                    "shares": p["shares"],
                    "profit": safe_round(profit),
                    "profit_pct": safe_round(net_ret * 100),
                    "exit_reason": exit_reason,
                })
                balance += p["shares"] * exit_price
                del pos[ticker]
            elif p["shares"] <= 0:
                del pos[ticker]

        # Enter new positions
        per_trade = balance * risk_pct / max_pos if max_pos > 0 else 0
        for ticker, price in candidates:
            if ticker in pos or len(pos) >= max_pos:
                continue
            entry_price = price * (1 + slippage_pct)
            shares = int(per_trade / entry_price) if entry_price > 0 else 0
            if shares < 1:
                continue
            cost = shares * entry_price * (1 + cost_pct)
            if cost > balance:
                continue
            balance -= cost
            atr_entry = float(ticker_data[ticker]["atr"][i_c]) if ticker in ticker_data and i_c < len(ticker_data[ticker]["atr"]) else 0
            pos[ticker] = {"entry_date": current_date, "entry_price": entry_price, "shares": shares,
                           "peak_price": entry_price, "entry_atr": max(atr_entry, price * 0.01), "trimmed": False}

        # Mark to market
        mtm = balance
        for t, p in pos.items():
            td_m = ticker_data.get(t)
            if td_m:
                i_m = int(np.searchsorted(td_m["dates"], current_date, side="right")) - 1
                if i_m >= 0:
                    mtm += p["shares"] * float(td_m["closes"][i_m])
        if mtm > peak:
            peak = mtm
        dd = (peak - mtm) / peak * 100 if peak > 0 else 0
        equity.append({
            "date": current_date.isoformat(),
            "balance": safe_round(mtm),
            "drawdown_pct": safe_round(dd),
        })

    # Close remaining positions
    if all_dates:
        last_date = all_dates[-1]
        for ticker, p in pos.items():
            exit_price = float(ticker_data[ticker]["closes"][-1]) if ticker in ticker_data else p["entry_price"]
            gross_ret = (exit_price - p["entry_price"]) / p["entry_price"]
            net_ret = gross_ret - 2 * cost_pct - slippage_pct
            profit = p["shares"] * (exit_price - p["entry_price"])
            trades.append({
                "ticker": ticker.replace(".NS", ""),
                "entry_date": p["entry_date"].isoformat(),
                "entry_price": safe_round(p["entry_price"]),
                "exit_date": last_date.isoformat(),
                "exit_price": safe_round(exit_price),
                "shares": p["shares"],
                "profit": safe_round(profit),
                "profit_pct": safe_round(net_ret * 100),
                "exit_reason": "End of period",
            })
            balance += p["shares"] * exit_price
        pos.clear()

    end_bal = balance
    for t, p in pos.items():
        if t in ticker_data:
            end_bal += p["shares"] * float(ticker_data[t]["closes"][-1])

    total_return = (end_bal - capital) / capital * 100 if capital > 0 else 0
    nifty_ret = (n_prices[-1] - n_prices[start_idx]) / n_prices[start_idx] * 100

    wins = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t["profit"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["profit"] for t in losses) / len(losses) if losses else 0

    # Sharpe from daily equity returns
    eq_balances = [e["balance"] for e in equity]
    if len(eq_balances) > 1:
        eq_rets = np.diff(eq_balances) / eq_balances[:-1]
        sharpe = float(eq_rets.mean() / eq_rets.std(ddof=0) * np.sqrt(252)) if eq_rets.std(ddof=0) > 0 else 0
    else:
        sharpe = 0

    max_dd = max(e["drawdown_pct"] for e in equity)

    nifty_eq = []
    for e in equity:
        # Find matching nifty price
        n_idx = len(n_prices) - len(equity) + equity.index(e)
        if 0 <= n_idx < len(n_prices):
            nv = n_prices[n_idx]
            nifty_ret_pct = (nv - n_prices[0]) / n_prices[0] * 100
            nifty_eq.append(nifty_ret_pct)

    return {
        "starting_capital": capital,
        "ending_capital": safe_round(end_bal),
        "total_profit": safe_round(end_bal - capital),
        "return_pct": safe_round(total_return),
        "nifty_return": safe_round(nifty_ret),
        "sharpe_ratio": safe_round(sharpe),
        "max_drawdown_pct": safe_round(max_dd),
        "win_rate": safe_round(win_rate),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "avg_win": safe_round(avg_win),
        "avg_loss": safe_round(avg_loss),
        "profit_factor": safe_round(abs(sum(t["profit"] for t in wins) / sum(abs(t["profit"]) for t in losses))) if losses and sum(abs(t["profit"]) for t in losses) > 0 else 0,
        "trades": trades,
        "equity_curve": equity,
        "nifty_curve": nifty_eq,
    }


def run_backtest_multi(params: dict) -> dict:
    """Run backtest for ALL strategies with shared params and return combined equity curves.
    Returns {strategies: [{strategyName, equity_curve, summary}, ...], starting_capital, nifty_curve}
    """
    base_params = {k: v for k, v in params.items() if k != "strategy"}
    strategy_names = [s for s in get_strategies() if s != "All"]

    lines = []
    for sname in strategy_names:
        sp = dict(base_params, strategy=sname)
        result = run_backtest(sp)
        lines.append({
            "strategyName": sname,
            "equity_curve": result.get("equity_curve", []),
            "summary": {
                "return_pct": result.get("return_pct", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                "win_rate": result.get("win_rate", 0),
                "total_trades": result.get("total_trades", 0),
                "ending_capital": result.get("ending_capital", 0),
            },
        })

    return {
        "strategies": lines,
        "starting_capital": params.get("starting_capital", 100000),
        "nifty_curve": lines[0].get("nifty_curve", []) if lines else [],
    }


def get_watchlist() -> list:
    """Return watchlist items from DB."""
    con = _conn()
    rows = con.execute("SELECT Ticker, EntryPrice FROM WatchlistItems").fetchall()
    con.close()
    return [{"ticker": r[0].replace(".NS", ""), "entry_price": safe_round(r[1])} for r in rows]


def add_to_watchlist(ticker: str, price: float) -> dict:
    """Add ticker to watchlist."""
    t = ticker + ".NS" if not ticker.endswith(".NS") else ticker
    con = _conn()
    try:
        con.execute("INSERT INTO WatchlistItems (Ticker, EntryPrice) VALUES (?, ?)", [t, price])
        con.commit()
    except Exception as e:
        con.close()
        return {"success": False, "error": str(e)}
    con.close()
    return {"success": True}


def remove_from_watchlist(ticker: str) -> dict:
    """Remove ticker from watchlist."""
    t = ticker + ".NS" if not ticker.endswith(".NS") else ticker
    con = _conn()
    try:
        con.execute("DELETE FROM WatchlistItems WHERE Ticker = ?", [t])
        con.commit()
    except Exception as e:
        con.close()
        return {"success": False, "error": str(e)}
    con.close()
    return {"success": True}


def sync_yahoo_data() -> dict:
    """Sync data from Yahoo Finance for all stocks (Nifty + all tracked stocks).
    THIS IS A LARGE OPERATION — downloads ~800 stocks.
    Returns status dict.
    """
    import yfinance as yf
    from time import sleep

    con = _conn()
    stocks = con.execute("SELECT Ticker FROM StockMetadatas").fetchall()
    tickers = [r[0] for r in stocks]
    con.close()

    synced = 0
    errors = []
    con = _conn()

    try:
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA cache_size=-8000")
    except Exception:
        pass

    for ticker in [NIFTY_TICKER] + tickers:
        try:
            data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if data.empty:
                continue
            # Extract single-level columns if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            for date, row in data.iterrows():
                dt = date.to_pydatetime()
                con.execute(
                    "INSERT OR REPLACE INTO DailyBars (Ticker, Date, Open, High, Low, Close, Volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [ticker, dt, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), int(row["Volume"])]
                )
            synced += 1
            sleep(0.3)  # rate limit
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            continue

    con.commit()
    con.close()
    return {"status": "completed", "synced": synced, "errors": errors, "total": len(tickers) + 1}


def sync_sector_data(period: str = "5y") -> dict:
    """Sync sector index data from Yahoo Finance for Rotation backtest.
    Downloads ~16 sector indices with 5 years of daily data.
    Returns status dict.
    """
    import yfinance as yf
    from time import sleep

    sector_tickers = [
        "^NSEI", "^NSEBANK", "^CNXAUTO", "^CNXIT", "^CNXPHARMA",
        "^CNXMETAL", "^CNXENERGY", "^CNXFMCG", "^CNXMEDIA", "^CNXREALTY",
        "^CNXPSUBANK", "^CNXINFRA",
        "NIFTY_FIN_SERVICE.NS",
        "NIFTY_OIL_AND_GAS.NS",
        "^CNXCONSUM",
    ]

    con = _conn()
    try:
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA cache_size=-8000")
    except Exception:
        pass

    synced = 0
    errors = []
    for ticker in sector_tickers:
        try:
            data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if data.empty:
                errors.append(f"{ticker}: no data returned")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            for date, row in data.iterrows():
                dt = date.to_pydatetime()
                con.execute(
                    "INSERT OR REPLACE INTO SectorDailyBars (Ticker, Date, Open, High, Low, Close, Volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [ticker, dt, float(row["Open"]), float(row["High"]),
                     float(row["Low"]), float(row["Close"]), int(row["Volume"])]
                )
            synced += 1
            sleep(0.3)
        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:80]}")
            continue

    con.commit()
    con.close()
    return {
        "status": "completed",
        "synced": synced,
        "errors": errors,
        "total": len(sector_tickers),
        "period": period,
    }


def export_sector_data_csv() -> str:
    """Export SectorDailyBars to CSV string for download."""
    import io
    con = _conn()
    rows = con.execute("""
        SELECT Ticker, Date, Open, High, Low, Close, Volume
        FROM SectorDailyBars
        ORDER BY Ticker, Date
    """).fetchall()
    con.close()
    if not rows:
        return ""
    buf = io.StringIO()
    buf.write("Ticker,Date,Open,High,Low,Close,Volume\n")
    for r in rows:
        dt = r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1])
        buf.write(f"{r[0]},{dt},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")
    return buf.getvalue()