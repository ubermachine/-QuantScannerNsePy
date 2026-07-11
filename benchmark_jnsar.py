import time
import numpy as np
from indicators import ema

def jnsar_orig(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    n = len(closes)
    out = np.zeros(n)
    if n < 15:
        return out
    he5, le5, ce5 = ema(highs, 5), ema(lows, 5), ema(closes, 5)
    for i in range(4, n):
        out[i] = round((he5[i - 4:i + 1].sum() + le5[i - 4:i + 1].sum() + ce5[i - 4:i + 1].sum()) / 15.0, 2)
    out[:4] = closes[:4]
    return out

def jnsar_vectorized(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    n = len(closes)
    out = np.zeros(n)
    if n < 15:
        return out
    he5, le5, ce5 = ema(highs, 5), ema(lows, 5), ema(closes, 5)

    total_ema = he5 + le5 + ce5
    rolling_sum = np.convolve(total_ema, np.ones(5), mode='valid')
    out[4:] = np.round(rolling_sum / 15.0, 2)
    out[:4] = closes[:4]
    return out

# Generate dummy data
N = 100000
closes = np.random.rand(N) * 100 + 50
highs = closes + np.random.rand(N) * 5
lows = closes - np.random.rand(N) * 5

# Benchmark
start = time.time()
jnsar_orig(closes, highs, lows)
orig_time = time.time() - start

start = time.time()
jnsar_vectorized(closes, highs, lows)
vect_time = time.time() - start

print(f"Original: {orig_time:.5f}s")
print(f"Vectorized: {vect_time:.5f}s")
print(f"Speedup: {orig_time / vect_time:.2f}x")
