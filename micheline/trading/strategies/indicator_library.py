"""
micheline/trading/strategies/indicator_library.py

Bibliothèque COMPLÈTE d'indicateurs techniques.
Inclut : Ichimoku, Volume Profile, ICT/SMC, SuperTrend, VWAP,
         Pivots, Fibonacci, Donchian, Keltner, Williams %R,
         CCI, MFI, OBV, et bien plus.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("micheline.trading.indicators")


# ═══════════════════════════════════════════════════════
# INDICATEURS DE BASE (améliorés)
# ═══════════════════════════════════════════════════════

def calc_sma(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    if period > n or period < 1:
        return result
    cumsum = np.cumsum(np.insert(data, 0, 0))
    result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    if period > n or period < 1:
        return result
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def calc_wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average."""
    n = len(data)
    result = np.full(n, np.nan)
    if period > n or period < 1:
        return result
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    for i in range(period - 1, n):
        result[i] = np.sum(data[i - period + 1:i + 1] * weights) / weight_sum
    return result


def calc_hma(data: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average — plus réactif, moins de lag."""
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma_half = calc_wma(data, half_period)
    wma_full = calc_wma(data, period)
    diff = 2 * wma_half - wma_full
    # Remplacer NaN par 0 pour le dernier WMA
    valid = ~np.isnan(diff)
    if np.sum(valid) < sqrt_period:
        return np.full(len(data), np.nan)
    result = calc_wma(np.nan_to_num(diff, nan=0.0), sqrt_period)
    # Restaurer NaN au début
    first_valid = np.where(valid)[0]
    if len(first_valid) > 0:
        for i in range(first_valid[0]):
            result[i] = np.nan
    return result


def calc_dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average."""
    ema1 = calc_ema(data, period)
    valid = ~np.isnan(ema1)
    if np.sum(valid) < period:
        return ema1
    ema2 = calc_ema(np.nan_to_num(ema1, nan=0.0), period)
    result = 2 * ema1 - ema2
    for i in range(len(data)):
        if np.isnan(ema1[i]) or np.isnan(ema2[i]):
            result[i] = np.nan
    return result


def calc_tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average."""
    ema1 = calc_ema(data, period)
    ema2 = calc_ema(np.nan_to_num(ema1, nan=0.0), period)
    ema3 = calc_ema(np.nan_to_num(ema2, nan=0.0), period)
    result = 3 * ema1 - 3 * ema2 + ema3
    for i in range(len(data)):
        if np.isnan(ema1[i]) or np.isnan(ema2[i]) or np.isnan(ema3[i]):
            result[i] = np.nan
    return result


# ═══════════════════════════════════════════════════════
# RSI et variantes
# ═══════════════════════════════════════════════════════

def calc_rsi(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    if period + 1 > n or period < 2:
        return result
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        result[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        idx = i - 1
        avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        if avg_loss == 0:
            result[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def calc_stoch_rsi(data: np.ndarray, rsi_period: int = 14,
                   stoch_period: int = 14, k_smooth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI — RSI du RSI avec fenêtre stochastique."""
    rsi = calc_rsi(data, rsi_period)
    n = len(data)
    stoch_rsi = np.full(n, np.nan)

    for i in range(rsi_period + stoch_period - 1, n):
        window = rsi[i - stoch_period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        high = np.max(valid)
        low = np.min(valid)
        if high != low:
            stoch_rsi[i] = ((rsi[i] - low) / (high - low)) * 100.0
        else:
            stoch_rsi[i] = 50.0

    k_line = calc_sma(np.nan_to_num(stoch_rsi, nan=50.0), k_smooth)
    d_line = calc_sma(np.nan_to_num(k_line, nan=50.0), k_smooth)
    return k_line, d_line


# ═══════════════════════════════════════════════════════
# MACD et variantes
# ═══════════════════════════════════════════════════════

def calc_macd(data: np.ndarray, fast: int = 12, slow: int = 26,
              signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    ema_fast = calc_ema(data, fast)
    ema_slow = calc_ema(data, slow)
    macd_line = ema_fast - ema_slow

    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full(n, np.nan)
    if np.sum(valid_mask) > signal:
        valid_data = macd_line[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        ema_signal = calc_ema(valid_data, signal)
        for j in range(len(ema_signal)):
            if not np.isnan(ema_signal[j]):
                signal_line[valid_indices[j]] = ema_signal[j]

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ═══════════════════════════════════════════════════════
# BOLLINGER BANDS et variantes
# ═══════════════════════════════════════════════════════

def calc_bollinger(data: np.ndarray, period: int = 20,
                   deviation: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    middle = calc_sma(data, period)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(data[i - period + 1:i + 1])
    upper = middle + deviation * std
    lower = middle - deviation * std
    return upper, middle, lower


def calc_bollinger_width(upper: np.ndarray, lower: np.ndarray,
                         middle: np.ndarray) -> np.ndarray:
    """Bollinger Band Width — mesure la volatilité."""
    with np.errstate(divide='ignore', invalid='ignore'):
        width = np.where(middle > 0, (upper - lower) / middle * 100, 0)
    return width


def calc_bollinger_pct_b(close: np.ndarray, upper: np.ndarray,
                         lower: np.ndarray) -> np.ndarray:
    """%B — position du prix dans les bandes (0=lower, 1=upper)."""
    band_width = upper - lower
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_b = np.where(band_width > 0, (close - lower) / band_width, 0.5)
    return pct_b


# ═══════════════════════════════════════════════════════
# STOCHASTIC
# ═══════════════════════════════════════════════════════

def calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    k_period: int = 14, d_period: int = 3,
                    slowing: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close)
    raw_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1:i + 1])
        lowest = np.min(low[i - k_period + 1:i + 1])
        if highest != lowest:
            raw_k[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0
        else:
            raw_k[i] = 50.0
    k_line = calc_sma(np.nan_to_num(raw_k, nan=50.0), max(1, slowing))
    for i in range(k_period - 1):
        k_line[i] = np.nan
    d_line = calc_sma(np.nan_to_num(k_line, nan=50.0), d_period)
    for i in range(min(k_period + slowing - 2, n)):
        d_line[i] = np.nan
    return k_line, d_line


# ═══════════════════════════════════════════════════════
# ATR / ADX
# ═══════════════════════════════════════════════════════

def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    if period <= n and period >= 1:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX avec +DI et -DI retournés."""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    atr = calc_atr(high, low, close, period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    smooth_plus = calc_ema(plus_dm, period)
    smooth_minus = calc_ema(minus_dm, period)

    safe_atr = np.where((~np.isnan(atr)) & (atr > 0), atr, 1.0)
    plus_di = np.where((~np.isnan(atr)) & (atr > 0), (smooth_plus / safe_atr) * 100, 0)
    minus_di = np.where((~np.isnan(atr)) & (atr > 0), (smooth_minus / safe_atr) * 100, 0)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    safe_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = np.where(di_sum > 0, (di_diff / safe_sum) * 100, 0)

    adx = calc_ema(dx, period)
    return adx, plus_di, minus_di


# ═══════════════════════════════════════════════════════
# 🏯 ICHIMOKU KINKO HYO
# ═══════════════════════════════════════════════════════

def calc_ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  tenkan_period: int = 9, kijun_period: int = 26,
                  senkou_b_period: int = 52,
                  displacement: int = 26) -> Dict[str, np.ndarray]:
    """
    Ichimoku Kinko Hyo complet.

    Retourne:
        tenkan_sen: Ligne de conversion (moyenne haute+basse sur tenkan_period)
        kijun_sen: Ligne de base (moyenne haute+basse sur kijun_period)
        senkou_span_a: Leading Span A (moyenne tenkan+kijun, projeté displacement barres)
        senkou_span_b: Leading Span B (moyenne haute+basse sur senkou_b_period, projeté)
        chikou_span: Lagging Span (close décalé de -displacement)
        kumo_top: Haut du nuage (max de span A et B)
        kumo_bottom: Bas du nuage (min de span A et B)
    """
    n = len(close)

    # Tenkan-sen (Conversion Line)
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i - tenkan_period + 1:i + 1]) +
                     np.min(low[i - tenkan_period + 1:i + 1])) / 2

    # Kijun-sen (Base Line)
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i - kijun_period + 1:i + 1]) +
                    np.min(low[i - kijun_period + 1:i + 1])) / 2

    # Senkou Span A (Leading Span A) — projeté displacement barres en avant
    span_a_raw = (tenkan + kijun) / 2
    senkou_a = np.full(n, np.nan)
    for i in range(n - displacement):
        if not np.isnan(span_a_raw[i]):
            senkou_a[i + displacement] = span_a_raw[i]

    # Senkou Span B (Leading Span B) — projeté displacement barres en avant
    span_b_raw = np.full(n, np.nan)
    for i in range(senkou_b_period - 1, n):
        span_b_raw[i] = (np.max(high[i - senkou_b_period + 1:i + 1]) +
                         np.min(low[i - senkou_b_period + 1:i + 1])) / 2
    senkou_b = np.full(n, np.nan)
    for i in range(n - displacement):
        if not np.isnan(span_b_raw[i]):
            senkou_b[i + displacement] = span_b_raw[i]

    # Chikou Span (Lagging Span) — close décalé de displacement en arrière
    chikou = np.full(n, np.nan)
    for i in range(displacement, n):
        chikou[i - displacement] = close[i]

    # Kumo (nuage)
    kumo_top = np.maximum(
        np.nan_to_num(senkou_a, nan=-np.inf),
        np.nan_to_num(senkou_b, nan=-np.inf)
    )
    kumo_bottom = np.minimum(
        np.nan_to_num(senkou_a, nan=np.inf),
        np.nan_to_num(senkou_b, nan=np.inf)
    )
    # Restaurer NaN
    both_nan = np.isnan(senkou_a) & np.isnan(senkou_b)
    kumo_top[both_nan] = np.nan
    kumo_bottom[both_nan] = np.nan

    return {
        "tenkan_sen": tenkan,
        "kijun_sen": kijun,
        "senkou_span_a": senkou_a,
        "senkou_span_b": senkou_b,
        "chikou_span": chikou,
        "kumo_top": kumo_top,
        "kumo_bottom": kumo_bottom,
    }


# ═══════════════════════════════════════════════════════
# 💰 ICT / SMC (Smart Money Concepts)
# ═══════════════════════════════════════════════════════

def calc_market_structure(high: np.ndarray, low: np.ndarray,
                          lookback: int = 5) -> Dict[str, np.ndarray]:
    """
    Détecte la structure de marché ICT/SMC :
    - Higher Highs / Higher Lows (tendance haussière)
    - Lower Highs / Lower Lows (tendance baissière)
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    """
    n = len(high)
    trend = np.zeros(n)  # 1=bullish, -1=bearish, 0=range
    bos = np.zeros(n)    # 1=bullish BOS, -1=bearish BOS
    choch = np.zeros(n)  # 1=bullish CHoCH, -1=bearish CHoCH

    # Détecter les swing highs et lows
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        # Swing High
        if high[i] == np.max(high[i - lookback:i + lookback + 1]):
            swing_highs[i] = high[i]
        # Swing Low
        if low[i] == np.min(low[i - lookback:i + lookback + 1]):
            swing_lows[i] = low[i]

    # Analyser la structure
    last_high = np.nan
    last_low = np.nan
    prev_high = np.nan
    prev_low = np.nan
    current_trend = 0

    for i in range(lookback, n):
        if not np.isnan(swing_highs[i]):
            prev_high = last_high
            last_high = swing_highs[i]
        if not np.isnan(swing_lows[i]):
            prev_low = last_low
            last_low = swing_lows[i]

        if not np.isnan(last_high) and not np.isnan(prev_high):
            if not np.isnan(last_low) and not np.isnan(prev_low):
                if last_high > prev_high and last_low > prev_low:
                    new_trend = 1  # Bullish
                elif last_high < prev_high and last_low < prev_low:
                    new_trend = -1  # Bearish
                else:
                    new_trend = current_trend

                # BOS et CHoCH
                if new_trend == current_trend:
                    if new_trend != 0:
                        bos[i] = new_trend
                elif new_trend != 0 and current_trend != 0:
                    choch[i] = new_trend

                current_trend = new_trend

        trend[i] = current_trend

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }


def calc_order_blocks(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      open_price: np.ndarray,
                      lookback: int = 10) -> Dict[str, np.ndarray]:
    """
    Détecte les Order Blocks ICT/SMC.
    Un OB bullish = dernière bougie baissière avant un mouvement haussier fort.
    Un OB bearish = dernière bougie haussière avant un mouvement baissier fort.
    """
    n = len(close)
    ob_bullish = np.zeros(n)  # 1 = OB bullish détecté
    ob_bearish = np.zeros(n)  # 1 = OB bearish détecté
    ob_bull_high = np.full(n, np.nan)
    ob_bull_low = np.full(n, np.nan)
    ob_bear_high = np.full(n, np.nan)
    ob_bear_low = np.full(n, np.nan)

    atr = calc_atr(high, low, close, 14)

    for i in range(lookback + 2, n):
        if np.isnan(atr[i]):
            continue
        threshold = atr[i] * 1.5

        # Mouvement haussier fort
        if close[i] - close[i - 3] > threshold:
            # Chercher la dernière bougie baissière avant le mouvement
            for j in range(i - 1, max(i - lookback, 0), -1):
                if close[j] < open_price[j]:  # Bougie baissière
                    ob_bullish[j] = 1
                    ob_bull_high[j] = high[j]
                    ob_bull_low[j] = low[j]
                    break

        # Mouvement baissier fort
        if close[i - 3] - close[i] > threshold:
            for j in range(i - 1, max(i - lookback, 0), -1):
                if close[j] > open_price[j]:  # Bougie haussière
                    ob_bearish[j] = 1
                    ob_bear_high[j] = high[j]
                    ob_bear_low[j] = low[j]
                    break

    return {
        "ob_bullish": ob_bullish,
        "ob_bearish": ob_bearish,
        "ob_bull_high": ob_bull_high,
        "ob_bull_low": ob_bull_low,
        "ob_bear_high": ob_bear_high,
        "ob_bear_low": ob_bear_low,
    }


def calc_fair_value_gaps(high: np.ndarray, low: np.ndarray,
                         close: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Détecte les Fair Value Gaps (FVG) ICT.
    FVG bullish = gap entre low[i+1] et high[i-1] (le prix n'a pas touché)
    FVG bearish = gap entre low[i-1] et high[i+1]
    """
    n = len(close)
    fvg_bull = np.zeros(n)
    fvg_bear = np.zeros(n)
    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_bottom = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_bottom = np.full(n, np.nan)

    for i in range(1, n - 1):
        # FVG Bullish: low de la bougie suivante > high de la bougie précédente
        if low[i + 1] > high[i - 1]:
            fvg_bull[i] = 1
            fvg_bull_top[i] = low[i + 1]
            fvg_bull_bottom[i] = high[i - 1]

        # FVG Bearish: high de la bougie suivante < low de la bougie précédente
        if high[i + 1] < low[i - 1]:
            fvg_bear[i] = 1
            fvg_bear_top[i] = low[i - 1]
            fvg_bear_bottom[i] = high[i + 1]

    return {
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "fvg_bull_top": fvg_bull_top,
        "fvg_bull_bottom": fvg_bull_bottom,
        "fvg_bear_top": fvg_bear_top,
        "fvg_bear_bottom": fvg_bear_bottom,
    }


def calc_liquidity_levels(high: np.ndarray, low: np.ndarray,
                          lookback: int = 20,
                          touch_count: int = 2) -> Dict[str, np.ndarray]:
    """
    Détecte les niveaux de liquidité (zones où le prix a été rejeté plusieurs fois).
    Ces zones attirent le prix (liquidity grab).
    """
    n = len(high)
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    liquidity_above = np.zeros(n)  # Liquidité au-dessus du prix
    liquidity_below = np.zeros(n)  # Liquidité en dessous du prix

    atr = calc_atr(high, low, np.full(n, (high + low) / 2), 14)

    for i in range(lookback * 2, n):
        if np.isnan(atr[i]):
            continue
        tolerance = atr[i] * 0.3

        # Chercher les niveaux de résistance
        recent_highs = high[i - lookback:i]
        for level in recent_highs:
            touches = np.sum(np.abs(high[i - lookback:i] - level) < tolerance)
            if touches >= touch_count:
                resistance[i] = level
                if level > high[i]:
                    liquidity_above[i] = 1
                break

        # Chercher les niveaux de support
        recent_lows = low[i - lookback:i]
        for level in recent_lows:
            touches = np.sum(np.abs(low[i - lookback:i] - level) < tolerance)
            if touches >= touch_count:
                support[i] = level
                if level < low[i]:
                    liquidity_below[i] = 1
                break

    return {
        "resistance": resistance,
        "support": support,
        "liquidity_above": liquidity_above,
        "liquidity_below": liquidity_below,
    }


# ═══════════════════════════════════════════════════════
# 📊 VOLUME PROFILE (approximé avec tick_volume)
# ═══════════════════════════════════════════════════════

def calc_volume_profile(close: np.ndarray, volume: np.ndarray,
                        lookback: int = 100,
                        num_levels: int = 50) -> Dict[str, np.ndarray]:
    """
    Volume Profile approximé.
    Calcule le POC (Point of Control), VAH, VAL.
    """
    n = len(close)
    poc = np.full(n, np.nan)     # Point of Control (prix avec le plus de volume)
    vah = np.full(n, np.nan)     # Value Area High
    val = np.full(n, np.nan)     # Value Area Low
    above_poc = np.zeros(n)      # Prix au-dessus du POC
    below_poc = np.zeros(n)      # Prix en dessous du POC

    for i in range(lookback, n):
        window_close = close[i - lookback:i]
        window_vol = volume[i - lookback:i].astype(float)

        if np.sum(window_vol) == 0:
            continue

        price_min = np.min(window_close)
        price_max = np.max(window_close)
        if price_max == price_min:
            continue

        # Créer des niveaux de prix
        levels = np.linspace(price_min, price_max, num_levels)
        vol_at_level = np.zeros(num_levels)

        for j in range(len(window_close)):
            idx = int((window_close[j] - price_min) / (price_max - price_min) * (num_levels - 1))
            idx = min(idx, num_levels - 1)
            vol_at_level[idx] += window_vol[j]

        # POC = niveau avec le plus de volume
        poc_idx = np.argmax(vol_at_level)
        poc[i] = levels[poc_idx]

        # Value Area (70% du volume)
        total_vol = np.sum(vol_at_level)
        target_vol = total_vol * 0.7
        cumul = vol_at_level[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while cumul < target_vol:
            expand_up = vol_at_level[high_idx + 1] if high_idx + 1 < num_levels else 0
            expand_down = vol_at_level[low_idx - 1] if low_idx - 1 >= 0 else 0

            if expand_up >= expand_down and high_idx + 1 < num_levels:
                high_idx += 1
                cumul += vol_at_level[high_idx]
            elif low_idx - 1 >= 0:
                low_idx -= 1
                cumul += vol_at_level[low_idx]
            else:
                break

        vah[i] = levels[high_idx]
        val[i] = levels[low_idx]

        above_poc[i] = 1 if close[i] > poc[i] else 0
        below_poc[i] = 1 if close[i] < poc[i] else 0

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "above_poc": above_poc,
        "below_poc": below_poc,
    }


# ═══════════════════════════════════════════════════════
# SUPERTREND
# ═══════════════════════════════════════════════════════

def calc_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int = 10, multiplier: float = 3.0) -> Dict[str, np.ndarray]:
    """SuperTrend — indicateur de tendance basé sur l'ATR."""
    n = len(close)
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)  # 1=up, -1=down

    for i in range(period, n):
        if np.isnan(atr[i]):
            continue

        basic_upper = hl2[i] + multiplier * atr[i]
        basic_lower = hl2[i] - multiplier * atr[i]

        if i == period:
            upper_band[i] = basic_upper
            lower_band[i] = basic_lower
            direction[i] = 1 if close[i] > basic_upper else -1
        else:
            # Upper band
            if not np.isnan(upper_band[i-1]):
                upper_band[i] = min(basic_upper, upper_band[i-1]) if close[i-1] <= upper_band[i-1] else basic_upper
            else:
                upper_band[i] = basic_upper

            # Lower band
            if not np.isnan(lower_band[i-1]):
                lower_band[i] = max(basic_lower, lower_band[i-1]) if close[i-1] >= lower_band[i-1] else basic_lower
            else:
                lower_band[i] = basic_lower

            # Direction
            if direction[i-1] == 1:
                direction[i] = 1 if close[i] >= lower_band[i] else -1
            else:
                direction[i] = -1 if close[i] <= upper_band[i] else 1

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return {
        "supertrend": supertrend,
        "direction": direction,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }


# ═══════════════════════════════════════════════════════
# DONCHIAN CHANNEL
# ═══════════════════════════════════════════════════════

def calc_donchian(high: np.ndarray, low: np.ndarray,
                  period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channel — breakout based."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)

    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2

    return upper, middle, lower


# ═══════════════════════════════════════════════════════
# KELTNER CHANNEL
# ═══════════════════════════════════════════════════════

def calc_keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 ema_period: int = 20, atr_period: int = 10,
                 multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel — EMA ± ATR * multiplier."""
    middle = calc_ema(close, ema_period)
    atr = calc_atr(high, low, close, atr_period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return upper, middle, lower


# ═══════════════════════════════════════════════════════
# WILLIAMS %R
# ═══════════════════════════════════════════════════════

def calc_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int = 14) -> np.ndarray:
    """Williams %R — oscillateur de momentum (-100 à 0)."""
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        if highest != lowest:
            result[i] = ((highest - close[i]) / (highest - lowest)) * -100
        else:
            result[i] = -50
    return result


# ═══════════════════════════════════════════════════════
# CCI (Commodity Channel Index)
# ═══════════════════════════════════════════════════════

def calc_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 20) -> np.ndarray:
    """CCI — mesure l'écart du prix par rapport à sa moyenne."""
    n = len(close)
    tp = (high + low + close) / 3
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        tp_window = tp[i - period + 1:i + 1]
        sma_tp = np.mean(tp_window)
        mean_dev = np.mean(np.abs(tp_window - sma_tp))
        if mean_dev > 0:
            result[i] = (tp[i] - sma_tp) / (0.015 * mean_dev)
        else:
            result[i] = 0

    return result


# ═══════════════════════════════════════════════════════
# MFI (Money Flow Index)
# ═══════════════════════════════════════════════════════

def calc_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period: int = 14) -> np.ndarray:
    """MFI — RSI pondéré par le volume."""
    n = len(close)
    tp = (high + low + close) / 3
    raw_mf = tp * volume.astype(float)
    result = np.full(n, np.nan)

    for i in range(period, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                if tp[j] > tp[j - 1]:
                    pos_flow += raw_mf[j]
                elif tp[j] < tp[j - 1]:
                    neg_flow += raw_mf[j]

        if neg_flow > 0:
            mfi_ratio = pos_flow / neg_flow
            result[i] = 100 - (100 / (1 + mfi_ratio))
        elif pos_flow > 0:
            result[i] = 100
        else:
            result[i] = 50

    return result


# ═══════════════════════════════════════════════════════
# OBV (On Balance Volume)
# ═══════════════════════════════════════════════════════

def calc_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On Balance Volume."""
    n = len(close)
    obv = np.zeros(n)
    vol_float = volume.astype(float)

    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol_float[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol_float[i]
        else:
            obv[i] = obv[i - 1]

    return obv


# ═══════════════════════════════════════════════════════
# PIVOT POINTS
# ═══════════════════════════════════════════════════════

def calc_pivot_points(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      period: int = 20) -> Dict[str, np.ndarray]:
    """
    Pivot Points classiques calculés sur une fenêtre glissante.
    PP = (H + L + C) / 3
    R1 = 2*PP - L, S1 = 2*PP - H
    R2 = PP + (H-L), S2 = PP - (H-L)
    R3 = H + 2*(PP-L), S3 = L - 2*(H-PP)
    """
    n = len(close)
    pp = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)

    for i in range(period, n):
        h = np.max(high[i - period:i])
        l = np.min(low[i - period:i])
        c = close[i - 1]

        pivot = (h + l + c) / 3
        pp[i] = pivot
        r1[i] = 2 * pivot - l
        s1[i] = 2 * pivot - h
        r2[i] = pivot + (h - l)
        s2[i] = pivot - (h - l)
        r3[i] = h + 2 * (pivot - l)
        s3[i] = l - 2 * (h - pivot)

    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


# ═══════════════════════════════════════════════════════
# FIBONACCI RETRACEMENT (dynamique)
# ═══════════════════════════════════════════════════════

def calc_fib_retracement(high: np.ndarray, low: np.ndarray,
                         lookback: int = 50) -> Dict[str, np.ndarray]:
    """
    Niveaux de Fibonacci dynamiques basés sur les swing récents.
    Niveaux: 0, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    """
    n = len(high)
    fib_levels = {
        "fib_0": np.full(n, np.nan),
        "fib_236": np.full(n, np.nan),
        "fib_382": np.full(n, np.nan),
        "fib_500": np.full(n, np.nan),
        "fib_618": np.full(n, np.nan),
        "fib_786": np.full(n, np.nan),
        "fib_100": np.full(n, np.nan),
    }

    for i in range(lookback, n):
        h = np.max(high[i - lookback:i])
        l = np.min(low[i - lookback:i])
        diff = h - l

        if diff == 0:
            continue

        fib_levels["fib_0"][i] = l
        fib_levels["fib_236"][i] = l + diff * 0.236
        fib_levels["fib_382"][i] = l + diff * 0.382
        fib_levels["fib_500"][i] = l + diff * 0.500
        fib_levels["fib_618"][i] = l + diff * 0.618
        fib_levels["fib_786"][i] = l + diff * 0.786
        fib_levels["fib_100"][i] = h

    return fib_levels


# ═══════════════════════════════════════════════════════
# VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════

def calc_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              volume: np.ndarray, period: int = 20) -> np.ndarray:
    """VWAP sur fenêtre glissante."""
    n = len(close)
    tp = (high + low + close) / 3
    vol_float = volume.astype(float)
    vwap = np.full(n, np.nan)

    for i in range(period, n):
        tp_window = tp[i - period:i + 1]
        vol_window = vol_float[i - period:i + 1]
        total_vol = np.sum(vol_window)
        if total_vol > 0:
            vwap[i] = np.sum(tp_window * vol_window) / total_vol

    return vwap


# ═══════════════════════════════════════════════════════
# MOMENTUM / ROC
# ═══════════════════════════════════════════════════════

def calc_momentum(data: np.ndarray, period: int = 10) -> np.ndarray:
    """Momentum = close[i] - close[i-period]."""
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period, n):
        result[i] = data[i] - data[i - period]
    return result


def calc_roc(data: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change = (close[i] - close[i-period]) / close[i-period] * 100."""
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period, n):
        if data[i - period] != 0:
            result[i] = (data[i] - data[i - period]) / data[i - period] * 100
    return result


# ═══════════════════════════════════════════════════════
# PARABOLIC SAR
# ═══════════════════════════════════════════════════════

def calc_parabolic_sar(high: np.ndarray, low: np.ndarray,
                       af_start: float = 0.02, af_step: float = 0.02,
                       af_max: float = 0.20) -> Dict[str, np.ndarray]:
    """Parabolic SAR — trailing stop dynamique."""
    n = len(high)
    sar = np.full(n, np.nan)
    direction = np.zeros(n)  # 1=long, -1=short

    # Initialisation
    af = af_start
    bull = True
    ep = high[0]
    sar[0] = low[0]
    direction[0] = 1

    for i in range(1, n):
        if bull:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], low[i-1])
            if i >= 2:
                sar[i] = min(sar[i], low[i-2])

            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)

            if low[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_start
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], high[i-1])
            if i >= 2:
                sar[i] = max(sar[i], high[i-2])

            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_start

        direction[i] = 1 if bull else -1

    return {"sar": sar, "direction": direction}