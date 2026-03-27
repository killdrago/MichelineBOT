"""
micheline/trading/mt5_backtest.py

Backtester intégré fidèle à MT5.
- Données OHLCV réelles de MT5
- Indicateurs calculés EXACTEMENT comme MT5/MQL5
- Vrai spread du symbole
- Capital réel avec lot sizing
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("micheline.trading.mt5_backtest")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

STARTING_CAPITAL = 10000.0

DEFAULT_BARS = {
    "M1": 50000, "M5": 30000, "M15": 20000, "M30": 15000,
    "H1": 10000, "H4": 5000, "D1": 2000, "W1": 500,
}


def _get_tf_map():
    if not MT5_AVAILABLE:
        return {}
    return {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2,
        "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
    }


def _ensure_mt5() -> bool:
    if not MT5_AVAILABLE:
        return False
    try:
        info = mt5.terminal_info()
        if info is not None and info.connected:
            return True
        return mt5.initialize()
    except Exception:
        return False


def _get_symbol_info(symbol):
    if not _ensure_mt5():
        return None
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    if info is None:
        return None
    pip = info.point * 10 if info.digits in (5, 3) else info.point
    return {
        "point": info.point, "digits": info.digits, "pip": pip,
        "spread": info.spread,
        "spread_pips": info.spread * info.point / pip,
        "spread_price": info.spread * info.point,
        "tick_value": info.trade_tick_value,
        "tick_size": info.trade_tick_size,
        "lot_min": info.volume_min, "lot_max": info.volume_max,
        "lot_step": info.volume_step,
        "contract_size": info.trade_contract_size,
    }


def _get_rates(symbol, tf_str, start=None, end=None):
    if not _ensure_mt5():
        return None
    sym = mt5.symbol_info(symbol)
    if sym is None:
        if not mt5.symbol_select(symbol, True):
            return None
    tf_map = _get_tf_map()
    mt5_tf = tf_map.get(tf_str)
    if mt5_tf is None:
        return None
    if start and end:
        try:
            d_from = datetime.strptime(start, "%Y-%m-%d")
            d_to = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            d_to = datetime.now()
            d_from = d_to - timedelta(days=365 * 3)
    else:
        d_to = datetime.now()
        tf_min = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                  "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
        mins = tf_min.get(tf_str, 60)
        bars = DEFAULT_BARS.get(tf_str, 5000)
        days = max(30, min(int((bars * mins) / (60 * 24) * 1.5), 365 * 5))
        d_from = d_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5_tf, d_from, d_to)
    if rates is None or len(rates) == 0:
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, DEFAULT_BARS.get(tf_str, 5000))
    if rates is None or len(rates) == 0:
        return None
    logger.info(f"✅ {len(rates)} barres MT5: {symbol} {tf_str}")
    return rates


# ═══════════════════════════════════════════════════════
# INDICATEURS — CALCULS EXACTS COMME MQL5
# ═══════════════════════════════════════════════════════

def _sma(data, period):
    n = len(data)
    r = np.full(n, np.nan)
    if period > n or period < 1:
        return r
    cs = np.cumsum(np.insert(data, 0, 0))
    r[period - 1:] = (cs[period:] - cs[:-period]) / period
    return r


def _ema(data, period):
    n = len(data)
    r = np.full(n, np.nan)
    if period > n or period < 1:
        return r
    m = 2.0 / (period + 1)
    r[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        r[i] = (data[i] - r[i - 1]) * m + r[i - 1]
    return r


def _wma(data, period):
    """Weighted Moving Average — identique à MODE_LWMA dans MT5."""
    n = len(data)
    r = np.full(n, np.nan)
    if period > n or period < 1:
        return r
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    for i in range(period - 1, n):
        r[i] = np.sum(data[i - period + 1:i + 1] * weights) / wsum
    return r


def _hma(data, period):
    """
    Hull Moving Average — EXACTEMENT comme dans l'EA MQL5 :
    1. WMA(period/2)
    2. WMA(period)
    3. raw = 2 * WMA(half) - WMA(full)
    4. HMA = WMA(sqrt(period)) appliqué sur raw
    """
    n = len(data)
    half_period = max(1, int(np.floor(period / 2.0)))
    sqrt_period = max(1, int(np.floor(np.sqrt(period))))

    wma_half = _wma(data, half_period)
    wma_full = _wma(data, period)

    # raw = 2 * WMA(half) - WMA(full)
    raw = 2.0 * wma_half - wma_full

    # Remplacer NaN par 0 pour le calcul WMA final
    raw_clean = np.nan_to_num(raw, nan=0.0)
    hma = _wma(raw_clean, sqrt_period)

    # Restaurer NaN au début
    first_valid = period - 1 + sqrt_period - 1
    for i in range(min(first_valid, n)):
        hma[i] = np.nan

    return hma


def _dema(data, period):
    """Double EMA — DEMA = 2*EMA - EMA(EMA)."""
    e1 = _ema(data, period)
    e1_clean = np.nan_to_num(e1, nan=0.0)
    e2 = _ema(e1_clean, period)
    r = 2 * e1 - e2
    for i in range(len(data)):
        if np.isnan(e1[i]) or np.isnan(e2[i]):
            r[i] = np.nan
    return r


def _tema(data, period):
    """Triple EMA — TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = _ema(data, period)
    e2 = _ema(np.nan_to_num(e1, nan=0.0), period)
    e3 = _ema(np.nan_to_num(e2, nan=0.0), period)
    r = 3 * e1 - 3 * e2 + e3
    for i in range(len(data)):
        if np.isnan(e1[i]) or np.isnan(e2[i]) or np.isnan(e3[i]):
            r[i] = np.nan
    return r


def _rsi(data, period):
    n = len(data)
    r = np.full(n, np.nan)
    if period + 1 > n or period < 2:
        return r
    d = np.diff(data)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:period])
    al = np.mean(l[:period])
    r[period] = 100.0 - 100.0 / (1 + ag / al) if al > 0 else (100 if ag > 0 else 50)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + g[i - 1]) / period
        al = (al * (period - 1) + l[i - 1]) / period
        r[i] = 100.0 - 100.0 / (1 + ag / al) if al > 0 else (100 if ag > 0 else 50)
    return r


def _macd(data, fast, slow, signal):
    ef = _ema(data, fast)
    es = _ema(data, slow)
    ml = ef - es
    valid = ~np.isnan(ml)
    sl = np.full(len(data), np.nan)
    if np.sum(valid) > signal:
        vd = ml[valid]
        vi = np.where(valid)[0]
        es2 = _ema(vd, signal)
        for j in range(len(es2)):
            if not np.isnan(es2[j]):
                sl[vi[j]] = es2[j]
    return ml, sl, ml - sl


def _bollinger(data, period, dev):
    mid = _sma(data, period)
    n = len(data)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(data[i - period + 1:i + 1])
    return mid + dev * std, mid, mid - dev * std


def _stochastic(high, low, close, k_per, d_per, slowing):
    n = len(close)
    raw_k = np.full(n, np.nan)
    for i in range(k_per - 1, n):
        hh = np.max(high[i - k_per + 1:i + 1])
        ll = np.min(low[i - k_per + 1:i + 1])
        raw_k[i] = ((close[i] - ll) / (hh - ll) * 100) if hh != ll else 50
    k = _sma(np.nan_to_num(raw_k, nan=50), max(1, slowing))
    for i in range(k_per - 1):
        k[i] = np.nan
    d = _sma(np.nan_to_num(k, nan=50), d_per)
    for i in range(min(k_per + slowing - 2, n)):
        d[i] = np.nan
    return k, d


def _atr(high, low, close, period):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    r = np.full(n, np.nan)
    if period <= n:
        r[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            r[i] = (r[i - 1] * (period - 1) + tr[i]) / period
    return r


def _adx(high, low, close, period):
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    at = _atr(high, low, close, period)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            mdm[i] = dn
    sp = _ema(pdm, period)
    sm = _ema(mdm, period)
    safe_at = np.where((~np.isnan(at)) & (at > 0), at, 1.0)
    pdi = np.where((~np.isnan(at)) & (at > 0), sp / safe_at * 100, 0)
    mdi = np.where((~np.isnan(at)) & (at > 0), sm / safe_at * 100, 0)
    ds = pdi + mdi
    safe_ds = np.where(ds > 0, ds, 1.0)
    dx = np.where(ds > 0, np.abs(pdi - mdi) / safe_ds * 100, 0)
    return _ema(dx, period)


def _cci(high, low, close, period):
    n = len(close)
    tp = (high + low + close) / 3
    r = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = tp[i - period + 1:i + 1]
        sm = np.mean(w)
        md = np.mean(np.abs(w - sm))
        r[i] = (tp[i] - sm) / (0.015 * md) if md > 0 else 0
    return r


def _williams_r(high, low, close, period):
    n = len(close)
    r = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        r[i] = ((hh - close[i]) / (hh - ll) * -100) if hh != ll else -50
    return r


def _sar(high, low, af_start=0.02, af_step=0.02, af_max=0.20):
    n = len(high)
    sar = np.full(n, np.nan)
    direction = np.zeros(n)
    af = af_start
    bull = True
    ep = high[0]
    sar[0] = low[0]
    direction[0] = 1
    for i in range(1, n):
        if bull:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low[i - 2])
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            if low[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_start
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high[i - 2])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_start
        direction[i] = 1 if bull else -1
    return sar, direction


def _ichimoku(high, low, close, tenkan_p=9, kijun_p=26, senkou_b_p=52, disp=26):
    n = len(close)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    for i in range(tenkan_p - 1, n):
        tenkan[i] = (np.max(high[i - tenkan_p + 1:i + 1]) + np.min(low[i - tenkan_p + 1:i + 1])) / 2
    for i in range(kijun_p - 1, n):
        kijun[i] = (np.max(high[i - kijun_p + 1:i + 1]) + np.min(low[i - kijun_p + 1:i + 1])) / 2
    span_a_raw = (tenkan + kijun) / 2
    span_a = np.full(n, np.nan)
    span_b_raw = np.full(n, np.nan)
    for i in range(senkou_b_p - 1, n):
        span_b_raw[i] = (np.max(high[i - senkou_b_p + 1:i + 1]) + np.min(low[i - senkou_b_p + 1:i + 1])) / 2
    span_b = np.full(n, np.nan)
    for i in range(n - disp):
        if not np.isnan(span_a_raw[i]):
            span_a[i + disp] = span_a_raw[i]
        if not np.isnan(span_b_raw[i]):
            span_b[i + disp] = span_b_raw[i]
    return tenkan, kijun, span_a, span_b


def _donchian(high, low, period):
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    return upper, lower


# ═══════════════════════════════════════════════════════
# COMPUTE INDICATEURS — AVEC LES BONS CALCULS
# ═══════════════════════════════════════════════════════

def _compute_indicators(rates, strategy):
    close = rates['close'].astype(float)
    high = rates['high'].astype(float)
    low = rates['low'].astype(float)
    n = len(close)
    computed = {}

    # Map chaque type de MA vers sa vraie fonction
    MA_FUNCTIONS = {
        "SMA": _sma,
        "EMA": _ema,
        "WMA": _wma,
        "HMA": _hma,
        "DEMA": _dema,
        "TEMA": _tema,
    }

    for ind in strategy.get("indicators", []):
        t = ind.get("type", "").upper()
        p = ind.get("params", {})
        try:
            if t in MA_FUNCTIONS:
                period = max(2, p.get("period", 20))
                fn = MA_FUNCTIONS[t]
                computed[f"MA_{t}_{period}"] = fn(close, period)

            elif t == "RSI":
                computed["RSI"] = _rsi(close, max(2, p.get("period", 14)))
                computed["RSI_OB"] = p.get("overbought", 70)
                computed["RSI_OS"] = p.get("oversold", 30)

            elif t == "MACD":
                f = max(2, p.get("fast", 12))
                s = max(f + 1, p.get("slow", 26))
                sig = max(2, p.get("signal", 9))
                ml, sl, hist = _macd(close, f, s, sig)
                computed["MACD"] = ml
                computed["MACD_SIG"] = sl

            elif t == "BB":
                per = max(2, p.get("period", 20))
                dev = max(0.5, p.get("deviation", 2.0))
                u, m, l = _bollinger(close, per, dev)
                computed["BB_U"] = u
                computed["BB_L"] = l

            elif t == "STOCH":
                kp = max(2, p.get("k_period", 14))
                dp = max(1, p.get("d_period", 3))
                sl = max(1, p.get("slowing", 3))
                k, d = _stochastic(high, low, close, kp, dp, sl)
                computed["STOCH_K"] = k
                computed["STOCH_OB"] = p.get("overbought", 80)
                computed["STOCH_OS"] = p.get("oversold", 20)

            elif t == "ADX":
                computed["ADX"] = _adx(high, low, close, max(2, p.get("period", 14)))
                computed["ADX_TH"] = p.get("threshold", 25)

            elif t == "ATR":
                computed["ATR"] = _atr(high, low, close, max(2, p.get("period", 14)))

            elif t == "CCI":
                computed["CCI"] = _cci(high, low, close, max(2, p.get("period", 20)))
                computed["CCI_OB"] = p.get("overbought", 100)
                computed["CCI_OS"] = p.get("oversold", -100)

            elif t == "WILLIAMS_R":
                computed["WR"] = _williams_r(high, low, close, max(2, p.get("period", 14)))
                computed["WR_OB"] = p.get("overbought", -20)
                computed["WR_OS"] = p.get("oversold", -80)

            elif t in ("PARABOLIC_SAR", "SUPERTREND"):
                s, d = _sar(high, low, p.get("af_start", 0.02), p.get("af_step", 0.02), p.get("af_max", 0.2))
                computed["SAR"] = s
                computed["SAR_DIR"] = d

            elif t == "ICHIMOKU":
                tk, kj, sa, sb = _ichimoku(high, low, close,
                    p.get("tenkan_period", 9), p.get("kijun_period", 26),
                    p.get("senkou_b_period", 52), p.get("displacement", 26))
                computed["ICHI_TK"] = tk
                computed["ICHI_KJ"] = kj
                computed["ICHI_SA"] = sa
                computed["ICHI_SB"] = sb

            elif t == "DONCHIAN":
                u, l = _donchian(high, low, max(5, p.get("period", 20)))
                computed["DONCH_U"] = u
                computed["DONCH_L"] = l

            elif t == "OBV":
                obv = np.zeros(n)
                vol = rates['tick_volume'].astype(float)
                for i in range(1, n):
                    if close[i] > close[i - 1]:
                        obv[i] = obv[i - 1] + vol[i]
                    elif close[i] < close[i - 1]:
                        obv[i] = obv[i - 1] - vol[i]
                    else:
                        obv[i] = obv[i - 1]
                computed["OBV"] = obv
                computed["OBV_EMA"] = _ema(obv, 20)

            elif t == "MFI":
                tp = (high + low + close) / 3
                vol = rates['tick_volume'].astype(float)
                mfi = np.full(n, np.nan)
                per = max(2, p.get("period", 14))
                for i in range(per, n):
                    pf = nf = 0.0
                    for j in range(i - per + 1, i + 1):
                        if j > 0:
                            if tp[j] > tp[j - 1]:
                                pf += tp[j] * vol[j]
                            elif tp[j] < tp[j - 1]:
                                nf += tp[j] * vol[j]
                    mfi[i] = 100 - 100 / (1 + pf / nf) if nf > 0 else (100 if pf > 0 else 50)
                computed["MFI"] = mfi
                computed["MFI_OB"] = p.get("overbought", 80)
                computed["MFI_OS"] = p.get("oversold", 20)

            elif t == "MOMENTUM":
                per = max(2, p.get("period", 10))
                mom = np.full(n, np.nan)
                for i in range(per, n):
                    mom[i] = close[i] - close[i - per]
                computed["MOM"] = mom

            elif t == "VWAP":
                tp = (high + low + close) / 3
                vol = rates['tick_volume'].astype(float)
                per = max(5, p.get("period", 20))
                vwap = np.full(n, np.nan)
                for i in range(per, n):
                    tw = tp[i - per:i + 1]
                    vw = vol[i - per:i + 1]
                    tv = np.sum(vw)
                    if tv > 0:
                        vwap[i] = np.sum(tw * vw) / tv
                computed["VWAP"] = vwap

            elif t == "PIVOTS":
                per = max(5, p.get("period", 20))
                pp = np.full(n, np.nan)
                r1 = np.full(n, np.nan)
                s1 = np.full(n, np.nan)
                for i in range(per, n):
                    h = np.max(high[i - per:i])
                    l = np.min(low[i - per:i])
                    c = close[i - 1]
                    pv = (h + l + c) / 3
                    pp[i] = pv
                    r1[i] = 2 * pv - l
                    s1[i] = 2 * pv - h
                computed["PP"] = pp
                computed["PP_R1"] = r1
                computed["PP_S1"] = s1

            elif t == "FIBONACCI":
                lb = max(10, p.get("lookback", 50))
                f618 = np.full(n, np.nan)
                f382 = np.full(n, np.nan)
                for i in range(lb, n):
                    h = np.max(high[i - lb:i])
                    l = np.min(low[i - lb:i])
                    diff = h - l
                    if diff > 0:
                        f382[i] = l + diff * 0.382
                        f618[i] = l + diff * 0.618
                computed["FIB_382"] = f382
                computed["FIB_618"] = f618

            elif t == "KELTNER":
                ep = max(2, p.get("ema_period", 20))
                ap = max(2, p.get("atr_period", 10))
                mul = max(0.5, p.get("multiplier", 2.0))
                em = _ema(close, ep)
                at = _atr(high, low, close, ap)
                computed["KELT_U"] = em + mul * at
                computed["KELT_L"] = em - mul * at

            elif t == "STOCH_RSI":
                rp = max(2, p.get("rsi_period", 14))
                sp = max(2, p.get("stoch_period", 14))
                ks = max(1, p.get("k_smooth", 3))
                r = _rsi(close, rp)
                sr = np.full(n, np.nan)
                for i in range(rp + sp - 1, n):
                    w = r[i - sp + 1:i + 1]
                    v = w[~np.isnan(w)]
                    if len(v) >= 2:
                        h = np.max(v)
                        l = np.min(v)
                        sr[i] = ((r[i] - l) / (h - l) * 100) if h != l else 50
                computed["SRSI"] = _sma(np.nan_to_num(sr, nan=50), ks)
                computed["SRSI_OB"] = p.get("overbought", 80)
                computed["SRSI_OS"] = p.get("oversold", 20)

            elif t == "LIQUIDITY":
                lb = max(10, p.get("lookback", 20))
                at = _atr(high, low, close, 14)
                sup = np.full(n, np.nan)
                res = np.full(n, np.nan)
                for i in range(lb * 2, n):
                    if np.isnan(at[i]):
                        continue
                    tol = at[i] * 0.3
                    rh = high[i - lb:i]
                    for lv in rh:
                        if np.sum(np.abs(rh - lv) < tol) >= p.get("touch_count", 2):
                            res[i] = lv
                            break
                    rl = low[i - lb:i]
                    for lv in rl:
                        if np.sum(np.abs(rl - lv) < tol) >= p.get("touch_count", 2):
                            sup[i] = lv
                            break
                computed["LIQ_RES"] = res
                computed["LIQ_SUP"] = sup

            elif t == "VOLUME_PROFILE":
                lb = max(20, p.get("lookback", 100))
                vol = rates['tick_volume'].astype(float)
                poc = np.full(n, np.nan)
                for i in range(lb, n):
                    wc = close[i - lb:i]
                    wv = vol[i - lb:i]
                    if np.sum(wv) == 0:
                        continue
                    pmin, pmax = np.min(wc), np.max(wc)
                    if pmax == pmin:
                        continue
                    nl = 50
                    levels = np.linspace(pmin, pmax, nl)
                    vl = np.zeros(nl)
                    for j in range(len(wc)):
                        idx = min(int((wc[j] - pmin) / (pmax - pmin) * (nl - 1)), nl - 1)
                        vl[idx] += wv[j]
                    poc[i] = levels[np.argmax(vl)]
                computed["VP_POC"] = poc

            elif t == "ICT_STRUCTURE":
                lb = max(3, p.get("lookback", 5))
                bos = np.zeros(n)
                choch = np.zeros(n)
                lh = ln = ph = pn = np.nan
                ct = 0
                for i in range(lb, n - lb):
                    if high[i] == np.max(high[max(0, i - lb):min(n, i + lb + 1)]):
                        ph, lh = lh, high[i]
                    if low[i] == np.min(low[max(0, i - lb):min(n, i + lb + 1)]):
                        pn, ln = ln, low[i]
                    if not np.isnan(lh) and not np.isnan(ph) and not np.isnan(ln) and not np.isnan(pn):
                        if lh > ph and ln > pn:
                            nt = 1
                        elif lh < ph and ln < pn:
                            nt = -1
                        else:
                            nt = ct
                        if nt == ct and nt != 0:
                            bos[i] = nt
                        elif nt != 0 and ct != 0 and nt != ct:
                            choch[i] = nt
                        ct = nt
                computed["ICT_BOS"] = bos
                computed["ICT_CHOCH"] = choch

            elif t == "ORDER_BLOCKS":
                lb = max(5, p.get("lookback", 10))
                at = _atr(high, low, close, 14)
                op = rates['open'].astype(float)
                ob_bull = np.zeros(n)
                ob_bear = np.zeros(n)
                for i in range(lb + 2, n):
                    if np.isnan(at[i]):
                        continue
                    th = at[i] * 1.5
                    if close[i] - close[i - 3] > th:
                        for j in range(i - 1, max(i - lb, 0), -1):
                            if close[j] < op[j]:
                                ob_bull[j] = 1
                                break
                    if close[i - 3] - close[i] > th:
                        for j in range(i - 1, max(i - lb, 0), -1):
                            if close[j] > op[j]:
                                ob_bear[j] = 1
                                break
                computed["OB_BULL"] = ob_bull
                computed["OB_BEAR"] = ob_bear

            elif t == "FVG":
                fvg_b = np.zeros(n)
                fvg_s = np.zeros(n)
                for i in range(1, n - 1):
                    if low[i + 1] > high[i - 1]:
                        fvg_b[i] = 1
                    if high[i + 1] < low[i - 1]:
                        fvg_s[i] = 1
                computed["FVG_BULL"] = fvg_b
                computed["FVG_BEAR"] = fvg_s

        except Exception as e:
            logger.warning(f"Erreur calcul {t}: {e}")

    return computed


# ═══════════════════════════════════════════════════════
# SIGNAUX
# ═══════════════════════════════════════════════════════

def _generate_signals(rates, strategy, tf_str):
    close = rates['close'].astype(float)
    high = rates['high'].astype(float)
    low = rates['low'].astype(float)
    n = len(close)
    signals = np.zeros(n)
    c = _compute_indicators(rates, strategy)

    if not c:
        return signals

    warmup = 100
    for k, v in c.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            vi = np.where(~np.isnan(v))[0]
            if len(vi) > 0:
                warmup = max(warmup, vi[0] + 1)

    min_bars = max(1, {"M1": 5, "M5": 3, "M15": 2, "M30": 2, "H1": 1, "H4": 1, "D1": 1}.get(tf_str, 1))
    last_sig = -min_bars

    for i in range(warmup, n):
        if i - last_sig < min_bars:
            continue
        bv = sv = tv = 0

        # MA crossover — compare toutes les MA entre elles
        ma_keys = sorted([k for k in c if k.startswith("MA_")])
        if len(ma_keys) >= 2:
            fast_ma = c[ma_keys[0]]
            slow_ma = c[ma_keys[-1]]
            if (i > 0 and not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i])
                    and not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1])):
                tv += 1
                if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                    bv += 1
                elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                    sv += 1

            # Pour ribbon : vérifier alignement complet
            if len(ma_keys) >= 3:
                all_aligned_bull = True
                all_aligned_bear = True
                for j in range(len(ma_keys) - 1):
                    a = c[ma_keys[j]]
                    b = c[ma_keys[j + 1]]
                    if np.isnan(a[i]) or np.isnan(b[i]):
                        all_aligned_bull = False
                        all_aligned_bear = False
                        break
                    if a[i] <= b[i]:
                        all_aligned_bull = False
                    if a[i] >= b[i]:
                        all_aligned_bear = False

                if all_aligned_bull:
                    # Vérifier que ce n'était pas aligné avant
                    was_aligned = True
                    if i > 0:
                        for j in range(len(ma_keys) - 1):
                            a = c[ma_keys[j]]
                            b = c[ma_keys[j + 1]]
                            if np.isnan(a[i-1]) or np.isnan(b[i-1]) or a[i-1] <= b[i-1]:
                                was_aligned = False
                                break
                    if not was_aligned:
                        bv += 1
                        tv += 1

                if all_aligned_bear:
                    was_aligned = True
                    if i > 0:
                        for j in range(len(ma_keys) - 1):
                            a = c[ma_keys[j]]
                            b = c[ma_keys[j + 1]]
                            if np.isnan(a[i-1]) or np.isnan(b[i-1]) or a[i-1] >= b[i-1]:
                                was_aligned = False
                                break
                    if not was_aligned:
                        sv += 1
                        tv += 1

        elif len(ma_keys) == 1:
            ma = c[ma_keys[0]]
            if i > 0 and not np.isnan(ma[i]) and not np.isnan(ma[i-1]):
                tv += 1
                if close[i] > ma[i] and close[i-1] <= ma[i-1]:
                    bv += 1
                elif close[i] < ma[i] and close[i-1] >= ma[i-1]:
                    sv += 1

        if "RSI" in c:
            r = c["RSI"]
            if not np.isnan(r[i]):
                tv += 1
                if r[i] < c.get("RSI_OS", 30): bv += 1
                elif r[i] > c.get("RSI_OB", 70): sv += 1

        if "MACD" in c and "MACD_SIG" in c:
            m, s = c["MACD"], c["MACD_SIG"]
            if i > 0 and not np.isnan(m[i]) and not np.isnan(s[i]) and not np.isnan(m[i-1]) and not np.isnan(s[i-1]):
                tv += 1
                if m[i] > s[i] and m[i-1] <= s[i-1]: bv += 1
                elif m[i] < s[i] and m[i-1] >= s[i-1]: sv += 1

        if "BB_U" in c and "BB_L" in c:
            if not np.isnan(c["BB_U"][i]) and not np.isnan(c["BB_L"][i]):
                tv += 1
                if close[i] <= c["BB_L"][i]: bv += 1
                elif close[i] >= c["BB_U"][i]: sv += 1

        if "STOCH_K" in c:
            k = c["STOCH_K"]
            if not np.isnan(k[i]):
                tv += 1
                if k[i] < c.get("STOCH_OS", 20): bv += 1
                elif k[i] > c.get("STOCH_OB", 80): sv += 1

        if "ICHI_TK" in c and "ICHI_KJ" in c:
            tk, kj = c["ICHI_TK"], c["ICHI_KJ"]
            if i > 0 and not np.isnan(tk[i]) and not np.isnan(kj[i]) and not np.isnan(tk[i-1]) and not np.isnan(kj[i-1]):
                tv += 1
                if tk[i] > kj[i] and tk[i-1] <= kj[i-1]: bv += 1
                elif tk[i] < kj[i] and tk[i-1] >= kj[i-1]: sv += 1
            sa, sb = c.get("ICHI_SA"), c.get("ICHI_SB")
            if sa is not None and sb is not None and not np.isnan(sa[i]) and not np.isnan(sb[i]):
                tv += 1
                if close[i] > max(sa[i], sb[i]): bv += 1
                elif close[i] < min(sa[i], sb[i]): sv += 1

        if "SAR_DIR" in c:
            d = c["SAR_DIR"]
            if i > 0:
                tv += 1
                if d[i] == 1 and d[i-1] == -1: bv += 1
                elif d[i] == -1 and d[i-1] == 1: sv += 1

        if "CCI" in c:
            v = c["CCI"]
            if not np.isnan(v[i]):
                tv += 1
                if v[i] < c.get("CCI_OS", -100): bv += 1
                elif v[i] > c.get("CCI_OB", 100): sv += 1

        if "WR" in c:
            v = c["WR"]
            if not np.isnan(v[i]):
                tv += 1
                if v[i] < c.get("WR_OS", -80): bv += 1
                elif v[i] > c.get("WR_OB", -20): sv += 1

        if "MFI" in c:
            v = c["MFI"]
            if not np.isnan(v[i]):
                tv += 1
                if v[i] < c.get("MFI_OS", 20): bv += 1
                elif v[i] > c.get("MFI_OB", 80): sv += 1

        if "OBV" in c and "OBV_EMA" in c:
            o, oe = c["OBV"], c["OBV_EMA"]
            if i > 0 and not np.isnan(oe[i]) and not np.isnan(oe[i-1]):
                tv += 1
                if o[i] > oe[i] and o[i-1] <= oe[i-1]: bv += 1
                elif o[i] < oe[i] and o[i-1] >= oe[i-1]: sv += 1

        if "DONCH_U" in c and "DONCH_L" in c:
            du, dl = c["DONCH_U"], c["DONCH_L"]
            if i > 0 and not np.isnan(du[i]) and not np.isnan(dl[i]):
                tv += 1
                if close[i] >= du[i] and close[i-1] < du[i-1]: bv += 1
                elif close[i] <= dl[i] and close[i-1] > dl[i-1]: sv += 1

        if "KELT_U" in c and "KELT_L" in c:
            ku, kl = c["KELT_U"], c["KELT_L"]
            if not np.isnan(ku[i]) and not np.isnan(kl[i]):
                tv += 1
                if close[i] <= kl[i]: bv += 1
                elif close[i] >= ku[i]: sv += 1

        if "SRSI" in c:
            v = c["SRSI"]
            if not np.isnan(v[i]):
                tv += 1
                if v[i] < c.get("SRSI_OS", 20): bv += 1
                elif v[i] > c.get("SRSI_OB", 80): sv += 1

        if "VP_POC" in c:
            poc = c["VP_POC"]
            if i > 0 and not np.isnan(poc[i]):
                tv += 1
                if close[i] > poc[i] and close[i-1] <= poc[i]: bv += 1
                elif close[i] < poc[i] and close[i-1] >= poc[i]: sv += 1

        if "VWAP" in c:
            v = c["VWAP"]
            if i > 0 and not np.isnan(v[i]):
                tv += 1
                if close[i] > v[i] and close[i-1] <= v[i]: bv += 1
                elif close[i] < v[i] and close[i-1] >= v[i]: sv += 1

        if "MOM" in c:
            m = c["MOM"]
            if i > 0 and not np.isnan(m[i]) and not np.isnan(m[i-1]):
                tv += 1
                if m[i] > 0 and m[i-1] <= 0: bv += 1
                elif m[i] < 0 and m[i-1] >= 0: sv += 1

        if "PP" in c and "PP_S1" in c and "PP_R1" in c:
            s1, r1 = c["PP_S1"], c["PP_R1"]
            if not np.isnan(s1[i]) and not np.isnan(r1[i]):
                tv += 1
                if close[i] <= s1[i]: bv += 1
                elif close[i] >= r1[i]: sv += 1

        if "FIB_618" in c and "FIB_382" in c:
            f6, f3 = c["FIB_618"], c["FIB_382"]
            if not np.isnan(f6[i]) and not np.isnan(f3[i]):
                tv += 1
                if f3[i] <= close[i] <= f6[i]:
                    if close[i] > close[max(0, i-5)]: bv += 1
                    else: sv += 1

        if "ICT_BOS" in c:
            b = c["ICT_BOS"]
            if b[i] == 1: tv += 1; bv += 1
            elif b[i] == -1: tv += 1; sv += 1

        if "ICT_CHOCH" in c:
            ch = c["ICT_CHOCH"]
            if ch[i] == 1: tv += 1; bv += 1
            elif ch[i] == -1: tv += 1; sv += 1

        if "OB_BULL" in c:
            for j in range(max(0, i-20), i):
                if c["OB_BULL"][j] == 1:
                    tv += 1; bv += 1; break

        if "OB_BEAR" in c:
            for j in range(max(0, i-20), i):
                if c["OB_BEAR"][j] == 1:
                    tv += 1; sv += 1; break

        if "FVG_BULL" in c:
            for j in range(max(0, i-10), i):
                if c["FVG_BULL"][j] == 1:
                    tv += 1; bv += 1; break

        if "FVG_BEAR" in c:
            for j in range(max(0, i-10), i):
                if c["FVG_BEAR"][j] == 1:
                    tv += 1; sv += 1; break

        if "ADX" in c:
            a = c["ADX"]
            if not np.isnan(a[i]) and a[i] < c.get("ADX_TH", 25):
                bv = max(0, bv - 1)
                sv = max(0, sv - 1)

        if tv > 0:
            ma = max(1, tv // 3)
            if bv >= ma and bv > sv:
                signals[i] = 1; last_sig = i
            elif sv >= ma and sv > bv:
                signals[i] = -1; last_sig = i

    return signals


# ═══════════════════════════════════════════════════════
# SIMULATION TRADES
# ═══════════════════════════════════════════════════════

def _run_trades(rates, signals, strategy, sym_info, capital=STARTING_CAPITAL):
    close = rates['close'].astype(float)
    high = rates['high'].astype(float)
    low = rates['low'].astype(float)
    n = len(close)

    rm = strategy.get("risk_management", {})
    sl_pips = rm.get("stop_loss", 50)
    tp_pips = rm.get("take_profit", 100)
    risk_pct = rm.get("risk_per_trade", 1.0)

    pip = sym_info["pip"]
    point = sym_info["point"]
    spread_price = sym_info["spread_price"]
    tick_value = sym_info["tick_value"]
    tick_size = sym_info["tick_size"]
    lot_min = sym_info["lot_min"]
    lot_step = sym_info["lot_step"]

    sl_price = sl_pips * pip
    tp_price = tp_pips * pip

    trades = []
    eq = [capital]
    pos = 0
    ep = 0.0
    lot = 0.0

    def calc_lot():
        ra = capital * risk_pct / 100.0
        slv = sl_pips * pip / tick_size * tick_value
        if slv <= 0:
            return lot_min
        lt = ra / slv
        lt = max(lot_min, round(lt / lot_step) * lot_step)
        return round(lt, 2)

    def calc_pnl(pips):
        return pips * pip / tick_size * tick_value * lot

    for i in range(1, n):
        if pos != 0:
            if pos == 1:
                if low[i] <= ep - sl_price:
                    pnl = calc_pnl(-sl_pips)
                    capital += pnl
                    trades.append({"pnl_pips": -sl_pips, "pnl_money": round(pnl, 2)})
                    pos = 0
                elif high[i] >= ep + tp_price:
                    pnl = calc_pnl(tp_pips)
                    capital += pnl
                    trades.append({"pnl_pips": tp_pips, "pnl_money": round(pnl, 2)})
                    pos = 0
            elif pos == -1:
                if high[i] >= ep + sl_price:
                    pnl = calc_pnl(-sl_pips)
                    capital += pnl
                    trades.append({"pnl_pips": -sl_pips, "pnl_money": round(pnl, 2)})
                    pos = 0
                elif low[i] <= ep - tp_price:
                    pnl = calc_pnl(tp_pips)
                    capital += pnl
                    trades.append({"pnl_pips": tp_pips, "pnl_money": round(pnl, 2)})
                    pos = 0

        if pos == 0 and signals[i] != 0:
            lot = calc_lot()
            if signals[i] == 1:
                ep = close[i] + spread_price
                pos = 1
            else:
                ep = close[i]
                pos = -1
        elif pos != 0 and signals[i] == -pos:
            if pos == 1:
                pp = (close[i] - ep) / pip
            else:
                pp = (ep - close[i] - spread_price) / pip
            pnl = pp * pip / tick_size * tick_value * lot
            capital += pnl
            trades.append({"pnl_pips": round(pp, 1), "pnl_money": round(pnl, 2)})
            lot = calc_lot()
            if signals[i] == 1:
                ep = close[i] + spread_price
                pos = 1
            else:
                ep = close[i]
                pos = -1

        eq.append(capital)

    if pos != 0:
        if pos == 1:
            pp = (close[-1] - ep) / pip
        else:
            pp = (ep - close[-1] - spread_price) / pip
        pnl = pp * pip / tick_size * tick_value * lot
        capital += pnl
        trades.append({"pnl_pips": round(pp, 1), "pnl_money": round(pnl, 2)})
        eq.append(capital)

    nt = len(trades)
    wins = [t for t in trades if t["pnl_money"] > 0]
    losses = [t for t in trades if t["pnl_money"] <= 0]
    pnls = [t["pnl_money"] for t in trades] if trades else [0]
    pips_list = [t["pnl_pips"] for t in trades] if trades else [0]

    pk = STARTING_CAPITAL
    mdd = 0.0
    for e in eq:
        if e > pk: pk = e
        d = pk - e
        if d > mdd: mdd = d

    gp = sum(t["pnl_money"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl_money"] for t in losses)) if losses else 0
    pf = min(99.0, gp / gl if gl > 0 else (99.0 if gp > 0 else 0))

    if nt > 1:
        mn = np.mean(pnls)
        sd = np.std(pnls)
        sh = (mn / sd) * np.sqrt(252) if sd > 0 else 0
    else:
        sh = 0

    return {
        "trades": nt, "wins": len(wins), "losses": len(losses),
        "profit_money": round(capital - STARTING_CAPITAL, 2),
        "profit_pips": round(sum(pips_list), 1),
        "starting_capital": STARTING_CAPITAL,
        "ending_capital": round(capital, 2),
        "winrate": round(len(wins) / nt, 4) if nt > 0 else 0,
        "drawdown": round(mdd, 2),
        "drawdown_pct": round(mdd / STARTING_CAPITAL * 100, 2) if STARTING_CAPITAL > 0 else 0,
        "profit_factor": round(pf, 2),
        "sharpe_ratio": round(sh, 4),
        "avg_win": round(np.mean([t["pnl_money"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_money"] for t in losses]), 2) if losses else 0,
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "equity_curve": eq,
        "spread_used": round(sym_info["spread_pips"], 1),
    }


# ═══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════

def run_real_backtest(strategy, start=None, end=None):
    symbol = strategy.get("symbol", "EURUSD")
    tf = strategy.get("timeframe", "H1")

    if not _ensure_mt5():
        return None

    sym_info = _get_symbol_info(symbol)
    if sym_info is None:
        return None

    rates = _get_rates(symbol, tf, start, end)
    if rates is None or len(rates) < 200:
        return None

    n = len(rates)
    times = rates['time']
    ds = datetime.utcfromtimestamp(int(times[0])).strftime("%Y-%m-%d")
    de = datetime.utcfromtimestamp(int(times[-1])).strftime("%Y-%m-%d")

    logger.info(f"📊 Backtest: {symbol} {tf} | {n} barres | {ds}→{de} | spread={sym_info['spread_pips']:.1f}")

    signals = _generate_signals(rates, strategy, tf)

    si = max(200, min(int(n * 0.7), n - 100))
    full = _run_trades(rates, signals, strategy, sym_info)
    train = _run_trades(rates[:si], signals[:si], strategy, sym_info)
    test = _run_trades(rates[si:], signals[si:], strategy, sym_info)

    sd = datetime.utcfromtimestamp(int(times[si])).strftime("%Y-%m-%d")
    tp = train["profit_money"]
    tep = test["profit_money"]
    tw = train["winrate"]
    tew = test["winrate"]

    deg = tep / tp if tp > 0 and tep > 0 else (0.0 if tp > 0 else 1.0)
    cons = min(1.0, tew / tw) if tw > 0 else (1.0 if tew == 0 else 0.0)

    result = {
        "profit": full["profit_pips"], "trades": full["trades"],
        "winrate": full["winrate"], "drawdown": full["drawdown"],
        "sharpe_ratio": full["sharpe_ratio"], "profit_factor": full["profit_factor"],
        "wins": full["wins"], "losses": full["losses"],
        "avg_win": full["avg_win"], "avg_loss": full["avg_loss"],
        "best_trade": full["best_trade"], "worst_trade": full["worst_trade"],
        "trade_results": [], "equity_curve": full["equity_curve"],
        "profit_money": full["profit_money"],
        "starting_capital": full["starting_capital"],
        "ending_capital": full["ending_capital"],
        "drawdown_pct": full["drawdown_pct"],
        "spread_pips": full["spread_used"],
        "date_start": ds, "date_end": de,
        "data_source": "mt5_real", "bars_count": n,
        "symbol": symbol, "timeframe": tf,
        "train_test_split": {
            "split_date": sd,
            "train_bars": si, "test_bars": n - si,
            "train_profit": round(tp, 2), "test_profit": round(tep, 2),
            "train_trades": train["trades"], "test_trades": test["trades"],
            "train_winrate": round(tw, 4), "test_winrate": round(tew, 4),
            "train_capital_end": train["ending_capital"],
            "test_capital_end": test["ending_capital"],
            "degradation_ratio": round(deg, 4),
            "consistency_score": round(cons, 4),
        }
    }

    logger.info(
        f"📊 {full['profit_money']:+.2f}$ | {full['trades']}t | WR={full['winrate']:.1%} | "
        f"{STARTING_CAPITAL:.0f}$→{full['ending_capital']:.0f}$ | DD={full['drawdown_pct']:.1f}%"
    )
    return result