"""Pure NumPy indicator functions — no classes, no state, returns arrays or scalars."""
import numpy as np
from typing import Tuple


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Full EMA array."""
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def ema_last(values: np.ndarray, period: int) -> float:
    """Only last EMA value (avoids full array alloc)."""
    if len(values) == 0:
        return 0.0
    k = 2.0 / (period + 1)
    e = float(values[0])
    for i in range(1, len(values)):
        e = float(values[i]) * k + e * (1 - k)
    return e


def multi_ema_last(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute ema8,10,21,50,200 in one pass. Returns (ema8, ema10, ema21, ema50, ema200)."""
    if len(values) == 0:
        return (0, 0, 0, 0, 0)
    k8, k10 = 2.0 / 9, 2.0 / 11
    k21, k50, k200 = 2.0 / 22, 2.0 / 51, 2.0 / 201
    e8 = e10 = e21 = e50 = e200 = float(values[0])
    for i in range(1, len(values)):
        v = float(values[i])
        e8 = v * k8 + e8 * (1 - k8)
        e10 = v * k10 + e10 * (1 - k10)
        e21 = v * k21 + e21 * (1 - k21)
        e50 = v * k50 + e50 * (1 - k50)
        e200 = v * k200 + e200 * (1 - k200)
    return (e8, e10, e21, e50, e200)


def jnsar(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """Custom JNSAR trailing stop indicator."""
    n = len(closes)
    out = np.zeros(n)
    if n < 15:
        return out
    he5, le5, ce5 = ema(highs, 5), ema(lows, 5), ema(closes, 5)
    total_sum = he5 + le5 + ce5
    r_sum = np.convolve(total_sum, np.ones(5), mode='valid')
    out[4:] = np.round(r_sum / 15.0, 2)
    out[:4] = closes[:4]
    return out


def jnsar_last(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
    """Only last JNSAR value."""
    n = len(closes)
    if n < 15:
        return float(closes[-1]) if n > 0 else 0.0
    return float(jnsar(closes, highs, lows)[-1])


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Full ATR array."""
    n = len(closes)
    out = np.zeros(n)
    if n < 2:
        return out
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - np.roll(closes, 1)),
                               np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    if n < period + 1:
        return out
    atr_val = float(tr[1:period + 1].sum() / period)
    out[period] = atr_val
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + float(tr[i])) / period
        out[i] = atr_val
    return out


def max_52_high(highs: np.ndarray) -> float:
    """Maximum value in the last ~250 bars."""
    start = max(0, len(highs) - 250)
    return float(highs[start:].max()) if len(highs) > start else 0.0


def swing_fib618(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Tuple[float, float, float]:
    """Fibonacci 61.8% retracement from swing high/low detection."""
    n = len(closes)
    if n < 10:
        return (0, 0, 0)
    sh, sl = float(highs[0]), float(lows[0])
    rsh, rsl = 0.0, 0.0
    state = "UP"
    for i in range(1, n):
        if state == "UP":
            if highs[i] > sh:
                sh = float(highs[i])
            elif closes[i] < sh * 0.95:
                rsh, state = sh, "DOWN"
                sl = float(lows[i])
        else:
            if lows[i] < sl:
                sl = float(lows[i])
            elif closes[i] > sl * 1.05:
                rsl, state = sl, "UP"
                sh = float(highs[i])
    if rsh == 0:
        rsh = float(highs.max())
    if rsl == 0:
        rsl = float(lows.min())
    if rsh <= rsl:
        rsh, rsl = float(highs.max()), float(lows.min())
    fib = rsh - 0.618 * (rsh - rsl)
    return (fib, rsh, rsl)


def volume_score(closes: np.ndarray, volumes: np.ndarray, lookback: int = 15) -> int:
    """Volume-weighted score 0-10."""
    if len(closes) < lookback + 1:
        return 0
    up = float(volumes[-lookback:][closes[-lookback:] > closes[-lookback - 1:-1]].sum())
    dn = float(volumes[-lookback:][closes[-lookback:] < closes[-lookback - 1:-1]].sum())
    total = up + dn
    return 5 if total == 0 else int(round(up / total * 10))


def relative_strength(s3m: float, i3m: float, s6m: float, i6m: float) -> float:
    """RS = excess return over index at 3M and 6M."""
    return (s3m - i3m) + (s6m - i6m)


def calc_return(closes: np.ndarray, days_ago: int) -> float:
    """Return % over N trading days."""
    if len(closes) < days_ago + 1:
        return 0.0
    return (float(closes[-1]) - float(closes[-days_ago - 1])) / float(closes[-days_ago - 1]) * 100.0


def volatility_fib_targets(closes: np.ndarray) -> Tuple[float, float, float]:
    """Target1, Target2, StopLoss from 10-day log-return volatility."""
    if len(closes) < 11:
        p = float(closes[-1])
        return (round(p * 1.05, 2), round(p * 1.10, 2), round(p * 0.95, 2))
    lr = np.log(closes[-10:] / closes[-11:-1])
    vol = float(np.std(lr, ddof=1))
    rng = float(closes[-1]) * vol * np.sqrt(10)
    p = float(closes[-1])
    return (round(p + 0.382 * rng, 2), round(p + 0.618 * rng, 2), round(p - 0.618 * rng, 2))


def macd(closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(macd_line, signal_line) arrays."""
    e12, e26 = ema(closes, 12), ema(closes, 26)
    macd_line = e12 - e26
    return (macd_line, ema(macd_line, 9))


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Full RSI array."""
    n = len(closes)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    changes = np.diff(closes)
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    avg_g = float(gains[:period].mean())
    avg_l = float(losses[:period].mean())
    out[period] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period + 1, n):
        g = gains[i - 1] if gains[i - 1] > 0 else 0
        l = losses[i - 1] if losses[i - 1] > 0 else 0
        avg_g = (avg_g * (period - 1) + float(g)) / period
        avg_l = (avg_l * (period - 1) + float(l)) / period
        out[i] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return out


def rsi_last(closes: np.ndarray, period: int = 14) -> float:
    """Only the last RSI value."""
    n = len(closes)
    if n <= period:
        return 50.0
    return float(rsi(closes, period)[-1])


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Full ADX array."""
    n = len(closes)
    out = np.zeros(n)
    if n < period * 2:
        return out
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    up = np.diff(highs)
    down = -np.diff(lows)
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    s_tr = float(tr[:period].sum())
    s_p = float(p_dm[:period].sum())
    s_m = float(m_dm[:period].sum())
    dx = np.zeros(n)
    for i in range(period, n):
        if i > period:
            idx = i - 1
            s_tr = s_tr - s_tr / period + float(tr[idx])
            s_p = s_p - s_p / period + float(p_dm[idx])
            s_m = s_m - s_m / period + float(m_dm[idx])
        pdi = 100 * s_p / s_tr if s_tr > 0 else 0
        mdi = 100 * s_m / s_tr if s_tr > 0 else 0
        di_sum = pdi + mdi
        dx[i] = 0 if di_sum == 0 else 100 * abs(pdi - mdi) / di_sum
    adx_val = float(dx[period:period * 2].sum() / period) if n >= period * 2 else 0
    out[period * 2 - 1] = adx_val
    for i in range(period * 2, n):
        adx_val = (adx_val * (period - 1) + float(dx[i])) / period
        out[i] = adx_val
    return out


def adx_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Only the last ADX value."""
    n = len(closes)
    if n < period * 2:
        return 0.0
    return float(adx(highs, lows, closes, period)[-1])


def bollinger(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(upper, middle, lower) arrays."""
    n = len(closes)
    u, m, lw = np.zeros(n), np.zeros(n), np.zeros(n)
    if n < period:
        return (u, m, lw)

    from numpy.lib.stride_tricks import sliding_window_view
    # Bolt Optimization: Replace Python loop with vectorized sliding_window_view
    windows = sliding_window_view(closes, period)
    mn = windows.mean(axis=-1)
    sd = windows.std(axis=-1, ddof=0)

    m[period - 1:] = mn
    u[period - 1:] = mn + mult * sd
    lw[period - 1:] = mn - mult * sd

    return (u, m, lw)


def bollinger_last(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> tuple[float, float, float]:
    """Only the last (upper, middle, lower) Bollinger values."""
    n = len(closes)
    if n < period:
        return (0.0, 0.0, 0.0)
    s = closes[-period:]
    mn = float(s.mean())
    sd = float(s.std(ddof=0))
    return (mn + mult * sd, mn, mn - mult * sd)


def cmf(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray,
        period: int = 21) -> np.ndarray:
    """Chaikin Money Flow array."""
    n = len(closes)
    out = np.zeros(n)
    if n < period:
        return out
    mfv = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-10) * volumes

    from numpy.lib.stride_tricks import sliding_window_view
    # Bolt Optimization: Replace Python loop with vectorized sliding_window_view
    mfv_windows = sliding_window_view(mfv, period)
    vol_windows = sliding_window_view(volumes, period)

    mfv_sum = mfv_windows.sum(axis=-1)
    vol_sum = vol_windows.sum(axis=-1)

    vol_sum_safe = np.where(vol_sum == 0, 1.0, vol_sum)
    res = mfv_sum / vol_sum_safe
    out[period - 1:] = np.where(vol_sum == 0, 0.0, res)
    return out


def cmf_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             volumes: np.ndarray, period: int = 21) -> float:
    """Only the last Chaikin Money Flow value."""
    n = len(closes)
    if n < period:
        return 0.0
    mfv = ((closes[-period:] - lows[-period:]) - (highs[-period:] - closes[-period:])) / (highs[-period:] - lows[-period:] + 1e-10) * volumes[-period:]
    vol_sum = float(volumes[-period:].sum())
    return float(mfv.sum()) / vol_sum if vol_sum > 0 else 0.0


def keltner(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
            period: int = 20, mult: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(upper, middle, lower) arrays."""
    n = len(closes)
    mid = ema(closes, period)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    if n > 1:
        tr[1:] = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - closes[:-1]),
                                       np.abs(lows[1:] - closes[:-1])))
    atr_arr = ema(tr, period)
    u = np.where(mid != 0, mid + mult * atr_arr, 0.0)
    lw = np.where(mid != 0, mid - mult * atr_arr, 0.0)
    return (u, mid, lw)


def keltner_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 20, mult: float = 1.5) -> tuple[float, float, float]:
    """Only the last (upper, middle, lower) Keltner channel values."""
    n = len(closes)
    if n < period:
        return (0.0, 0.0, 0.0)
    u, mid, lw = keltner(highs, lows, closes, period, mult)
    return (float(u[-1]), float(mid[-1]), float(lw[-1]))


def z_score_last(closes: np.ndarray, period: int = 50) -> float:
    """Z-score of last close relative to N-period window."""
    if len(closes) < period:
        return 0.0
    s = closes[-period:]
    std = float(s.std(ddof=0))
    return 0.0 if std == 0 else (float(s[-1]) - float(s.mean())) / std


def point_of_control(closes: np.ndarray, volumes: np.ndarray, lookback: int = 150) -> float:
    """Volume-weighted price center (simplified POC)."""
    if len(closes) == 0:
        return 0.0
    s = slice(max(0, len(closes) - lookback), len(closes))
    w = volumes[s]
    if w.sum() == 0:
        return float(closes[-1])
    return float(np.average(closes[s], weights=w))


def ytd_vwap(closes: np.ndarray, dates: np.ndarray, volumes: np.ndarray) -> float:
    """Year-to-date VWAP using only this year's data."""
    if len(closes) == 0:
        return 0.0
    try:
        yr = dates[-1].year if hasattr(dates[-1], 'year') else 2025
    except (IndexError, AttributeError):
        yr = 2025
    mask = np.array([getattr(d, 'year', yr) == yr for d in dates])
    if not mask.any():
        return float(closes[-1])
    w = volumes[mask]
    if w.sum() == 0 or not np.isfinite(w).all():
        return float(closes[-1])
    return float(np.average(closes[mask], weights=w))


def chandelier_exit(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 22, mult: float = 3.0) -> np.ndarray:
    """Chandelier Exit (long) trailing stop."""
    n = len(closes)
    out = np.zeros(n)
    if n < period:
        return out
    atr_arr = atr(highs, lows, closes, period)
    from numpy.lib.stride_tricks import sliding_window_view
    high_windows = sliding_window_view(highs, period)
    highest = high_windows.max(axis=-1)
    out[period:] = highest[1:] - mult * atr_arr[period:]
    return out


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume array."""
    n = len(closes)
    out = np.zeros(n)
    if n == 0:
        return out
    out[0] = float(volumes[0])
    if n < 2:
        return out
    diff = np.diff(closes)
    direction = np.zeros(n - 1)
    direction[diff > 0] = 1.0
    direction[diff < 0] = -1.0
    out[1:] = np.cumsum(direction * volumes[1:]) + out[0]
    return out


def rolling_sharpe(closes: np.ndarray, period: int = 60) -> float:
    """Rolling Sharpe ratio over last N days (annualized)."""
    if len(closes) < period + 1:
        return 0.0
    rets = closes[-period:] / closes[-period - 1:-1] - 1.0
    return 0.0 if float(rets.std(ddof=0)) == 0 else float(rets.mean() / rets.std(ddof=0) * np.sqrt(252))


def vol_percentile_rank(atr_arr: np.ndarray, lookback: int = 250) -> float:
    """Percentile rank of last ATR value within lookback window."""
    if len(atr_arr) < lookback:
        return 50.0
    s = atr_arr[-lookback:]
    last_val = float(s[-1])
    count = float((s < last_val).sum())
    return count / lookback * 100.0


def safe_round(v: float, decimals: int = 2) -> float:
    """NaN-safe rounding."""
    return round(v, decimals) if not (np.isnan(v) or np.isinf(v)) else 0.0


# --- Daily-chart strategy helpers ---

def j10sar(closes: np.ndarray) -> tuple:
    """J10SAR: EMA(10) mid-line + 2.5% envelope bands.
    Returns (mid_line, upper_band, lower_band) as floats."""
    mid = ema_last(closes, 10)
    return (mid, mid * 1.025, mid * 0.975)


def daily_swing_fib618(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                       lookback: int = 60) -> tuple:
    """61.8% retracement from swing high/low over a rolling lookback window.
    Returns (fib_618, swing_high, swing_low) as floats."""
    n = len(closes)
    if n < 10:
        return (0, 0, 0)
    start = max(0, n - lookback)
    c, h, l = closes[start:], highs[start:], lows[start:]
    sh, sl = float(h[0]), float(l[0])
    rsh, rsl = 0.0, 0.0
    state = "UP"
    for i in range(1, len(c)):
        if state == "UP":
            if h[i] > sh:
                sh = float(h[i])
            elif c[i] < sh * 0.95:
                rsh, state = sh, "DOWN"
                sl = float(l[i])
        else:
            if l[i] < sl:
                sl = float(l[i])
            elif c[i] > sl * 1.05:
                rsl, state = sl, "UP"
                sh = float(h[i])
    if rsh == 0:
        rsh = float(h.max())
    if rsl == 0:
        rsl = float(l.min())
    if rsh <= rsl:
        rsh, rsl = float(h.max()), float(l.min())
    fib = rsh - 0.618 * (rsh - rsl)
    return (fib, rsh, rsl)