"""
micheline/trading/mt5_backtest.py

Moteur de backtest RÉEL utilisant les données MT5.
Calcule les vrais indicateurs (RSI, EMA, SMA, MACD, STOCH, BB, ATR, ADX)
sur les vraies bougies et simule les trades avec SL/TP réels.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("micheline.trading.mt5_backtest")

# ═══════════════════════════════════════════════════════
# IMPORT MT5
# ═══════════════════════════════════════════════════════

MT5_AVAILABLE = False
mt5 = None
try:
    import MetaTrader5 as _mt5
    mt5 = _mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 non disponible pour le backtest réel")

# Cache des noms de symboles réels MT5
_symbol_cache = {}


# ═══════════════════════════════════════════════════════
# RÉSOLUTION AUTOMATIQUE DES SYMBOLES
# ═══════════════════════════════════════════════════════

def resolve_symbol(symbol: str) -> Optional[str]:
    """
    Trouve le vrai nom du symbole dans MT5.
    EURUSD → EURUSD ou EURUSD.raw ou EURUSDm etc.
    """
    if not MT5_AVAILABLE:
        return None

    # Cache
    if symbol in _symbol_cache:
        return _symbol_cache[symbol]

    # Test direct
    info = mt5.symbol_info(symbol)
    if info is not None:
        _symbol_cache[symbol] = symbol
        return symbol

    # Essayer des variantes
    suffixes = [".raw", ".pro", ".std", ".ecn", "m", ".m", "_m", ".i", ".s", ""]
    for suffix in suffixes:
        alt = symbol + suffix
        info = mt5.symbol_info(alt)
        if info is not None:
            _symbol_cache[symbol] = alt
            logger.info(f"Symbole résolu: {symbol} → {alt}")
            return alt

    # Recherche dans tous les symboles
    try:
        all_symbols = mt5.symbols_get()
        if all_symbols:
            symbol_upper = symbol.upper()
            for s in all_symbols:
                if symbol_upper in s.name.upper():
                    _symbol_cache[symbol] = s.name
                    logger.info(f"Symbole trouvé: {symbol} → {s.name}")
                    return s.name
    except Exception:
        pass

    logger.error(f"Symbole {symbol} introuvable dans MT5")
    return None


# ═══════════════════════════════════════════════════════
# CALCUL DES INDICATEURS
# ═══════════════════════════════════════════════════════

def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    ema = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


def calc_sma(data: np.ndarray, period: int) -> np.ndarray:
    sma = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return sma
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma


def calc_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.full_like(data, np.nan, dtype=float)
    if len(data) < period + 1:
        return rsi

    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calc_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = calc_ema(data, fast)
    ema_slow = calc_ema(data, slow)
    macd_line = ema_fast - ema_slow

    signal_line = np.full_like(data, np.nan, dtype=float)
    valid_macd = ~np.isnan(macd_line)
    if np.sum(valid_macd) >= signal:
        valid_indices = np.where(valid_macd)[0]
        first_valid = valid_indices[0]
        macd_values = macd_line[first_valid:]
        ema_sig = calc_ema(macd_values, signal)
        signal_line[first_valid:] = ema_sig

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     k_period: int = 14, d_period: int = 3, slowing: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close)
    raw_k = np.full(n, np.nan, dtype=float)
    stoch_k = np.full(n, np.nan, dtype=float)
    stoch_d = np.full(n, np.nan, dtype=float)

    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1:i + 1])
        lowest = np.min(low[i - k_period + 1:i + 1])
        if highest != lowest:
            raw_k[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0
        else:
            raw_k[i] = 50.0

    for i in range(k_period - 1 + slowing - 1, n):
        vals = raw_k[i - slowing + 1:i + 1]
        if not np.any(np.isnan(vals)):
            stoch_k[i] = np.mean(vals)

    for i in range(k_period - 1 + slowing - 1 + d_period - 1, n):
        vals = stoch_k[i - d_period + 1:i + 1]
        if not np.any(np.isnan(vals)):
            stoch_d[i] = np.mean(vals)

    return stoch_k, stoch_d


def calc_bollinger_bands(data: np.ndarray, period: int = 20, deviation: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    middle = calc_sma(data, period)
    upper = np.full_like(data, np.nan, dtype=float)
    lower = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        std = np.std(data[i - period + 1:i + 1])
        upper[i] = middle[i] + deviation * std
        lower[i] = middle[i] - deviation * std

    return upper, middle, lower


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=float)
    atr = np.full(n, np.nan, dtype=float)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    adx = np.full(n, np.nan, dtype=float)

    if n < period * 2:
        return adx

    atr = calc_atr(high, low, close, period)

    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    smooth_plus = np.full(n, np.nan, dtype=float)
    smooth_minus = np.full(n, np.nan, dtype=float)

    if n > period:
        smooth_plus[period] = np.sum(plus_dm[1:period + 1])
        smooth_minus[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            smooth_plus[i] = smooth_plus[i - 1] - (smooth_plus[i - 1] / period) + plus_dm[i]
            smooth_minus[i] = smooth_minus[i - 1] - (smooth_minus[i - 1] / period) + minus_dm[i]

    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    dx = np.full(n, np.nan, dtype=float)

    for i in range(period, n):
        if not np.isnan(atr[i]) and atr[i] > 0 and not np.isnan(smooth_plus[i]):
            plus_di[i] = (smooth_plus[i] / atr[i]) * 100
            minus_di[i] = (smooth_minus[i] / atr[i]) * 100
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100

    start_idx = period * 2
    if n > start_idx:
        valid_dx = []
        for i in range(period, n):
            if not np.isnan(dx[i]):
                valid_dx.append((i, dx[i]))
                if len(valid_dx) >= period:
                    break

        if len(valid_dx) >= period:
            first_adx_idx = valid_dx[-1][0]
            adx[first_adx_idx] = np.mean([v[1] for v in valid_dx])
            for i in range(first_adx_idx + 1, n):
                if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


# ═══════════════════════════════════════════════════════
# RÉCUPÉRATION DES DONNÉES MT5
# ═══════════════════════════════════════════════════════

def get_mt5_timeframe(tf_str: str) -> int:
    """
    Convertit un string timeframe en valeur numérique.
    Utilise les VALEURS directes au lieu des constantes MT5
    pour éviter les problèmes de thread.
    """
    mapping = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 16385,
        "H4": 16388,
        "D1": 16408,
        "W1": 32769,
        "MN1": 49153,
    }
    return mapping.get(tf_str.upper(), 16385)


def _ensure_mt5_init() -> bool:
    """
    Initialise MT5 de manière robuste.
    Gère les problèmes de thread en faisant plusieurs tentatives.
    """
    if not MT5_AVAILABLE:
        return False

    import time

    # Essayer 3 fois avec un petit délai
    for attempt in range(3):
        try:
            # Tenter de shutdown d'abord si déjà init dans un autre état
            if attempt > 0:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(0.5)

            if mt5.initialize():
                return True
            else:
                error = mt5.last_error()
                logger.warning(f"MT5 init tentative {attempt + 1}/3 échouée: {error}")

        except Exception as e:
            logger.warning(f"MT5 init exception tentative {attempt + 1}/3: {e}")

    logger.error("MT5 init échoué après 3 tentatives")
    return False


def fetch_mt5_data(symbol: str, timeframe: str, bars: int = 10000,
                    start_date: datetime = None, end_date: datetime = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Récupère les données OHLCV depuis MT5.
    Initialise MT5 de manière robuste à chaque appel.
    Utilise les valeurs numériques pour les timeframes.
    """
    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 non installé")
        return None

    # Initialisation robuste
    if not _ensure_mt5_init():
        logger.error("Impossible d'initialiser MT5")
        return None

    # Résoudre le nom du symbole
    real_symbol = resolve_symbol(symbol)
    if real_symbol is None:
        logger.error(f"Symbole {symbol} introuvable dans MT5")
        return None

    # S'assurer que le symbole est visible
    info = mt5.symbol_info(real_symbol)
    if info is not None and not info.visible:
        mt5.symbol_select(real_symbol, True)
        import time
        time.sleep(0.1)  # Petit délai pour laisser MT5 activer le symbole

    # Timeframe en valeur numérique
    mt5_tf = get_mt5_timeframe(timeframe)

    # Limiter à 50000 barres max
    bars = min(bars, 50000)

    # Récupérer les données
    rates = None
    try:
        if start_date and end_date:
            rates = mt5.copy_rates_range(real_symbol, mt5_tf, start_date, end_date)
        else:
            rates = mt5.copy_rates_from_pos(real_symbol, mt5_tf, 0, bars)
    except Exception as e:
        logger.error(f"Exception MT5 copy_rates: {e}")

    # Vérifier le résultat
    if rates is None or len(rates) < 50:
        error = mt5.last_error()
        actual_count = len(rates) if rates is not None else 0
        logger.error(f"Données insuffisantes {real_symbol} {timeframe}: "
                     f"{actual_count} barres (erreur MT5: {error})")

        # Retry avec moins de barres si c'était un problème de taille
        if bars > 5000:
            logger.info(f"Retry avec {bars // 2} barres...")
            try:
                rates = mt5.copy_rates_from_pos(real_symbol, mt5_tf, 0, bars // 2)
            except Exception:
                pass

        if rates is None or len(rates) < 50:
            return None

    logger.info(f"📊 Données MT5: {real_symbol} {timeframe} — {len(rates)} barres")

    # Convertir en arrays numpy
    data = {
        "open": np.array([r[1] for r in rates], dtype=float),
        "high": np.array([r[2] for r in rates], dtype=float),
        "low": np.array([r[3] for r in rates], dtype=float),
        "close": np.array([r[4] for r in rates], dtype=float),
        "volume": np.array([r[5] for r in rates], dtype=float),
        "time": np.array([r[0] for r in rates], dtype=int),
        "symbol": real_symbol,
        "point": info.point if info else 0.00001,
        "digits": info.digits if info else 5,
    }

    return data


# ═══════════════════════════════════════════════════════
# CALCUL DES SIGNAUX
# ═══════════════════════════════════════════════════════

def compute_indicators(data: Dict[str, np.ndarray], strategy: Dict[str, Any]) -> Dict[str, np.ndarray]:
    close = data["close"]
    high = data["high"]
    low = data["low"]
    indicators = strategy.get("indicators", [])
    computed = {}

    for ind in indicators:
        ind_type = ind.get("type", "").upper()
        params = ind.get("params", {})

        try:
            if ind_type == "RSI":
                period = params.get("period", 14)
                computed["RSI"] = calc_rsi(close, period)
                computed["RSI_overbought"] = params.get("overbought", 70)
                computed["RSI_oversold"] = params.get("oversold", 30)

            elif ind_type == "EMA":
                period = params.get("period", 20)
                computed[f"EMA_{period}"] = calc_ema(close, period)

            elif ind_type == "SMA":
                period = params.get("period", 50)
                computed[f"SMA_{period}"] = calc_sma(close, period)

            elif ind_type == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                macd_line, signal_line, histogram = calc_macd(close, fast, slow, signal)
                computed["MACD_main"] = macd_line
                computed["MACD_signal"] = signal_line
                computed["MACD_hist"] = histogram

            elif ind_type == "STOCH":
                k = params.get("k_period", 14)
                d = params.get("d_period", 3)
                slowing = params.get("slowing", 3)
                stoch_k, stoch_d = calc_stochastic(high, low, close, k, d, slowing)
                computed["STOCH_K"] = stoch_k
                computed["STOCH_D"] = stoch_d
                computed["STOCH_overbought"] = params.get("overbought", 80)
                computed["STOCH_oversold"] = params.get("oversold", 20)

            elif ind_type == "BB":
                period = params.get("period", 20)
                dev = params.get("deviation", 2.0)
                upper, middle, lower = calc_bollinger_bands(close, period, dev)
                computed["BB_upper"] = upper
                computed["BB_middle"] = middle
                computed["BB_lower"] = lower

            elif ind_type == "ATR":
                period = params.get("period", 14)
                computed["ATR"] = calc_atr(high, low, close, period)

            elif ind_type == "ADX":
                period = params.get("period", 14)
                computed["ADX"] = calc_adx(high, low, close, period)
                computed["ADX_threshold"] = params.get("threshold", 25)

        except Exception as e:
            logger.warning(f"Erreur calcul {ind_type}: {e}")

    return computed


def generate_signals(data: Dict[str, np.ndarray], indicators: Dict[str, np.ndarray],
                      strategy: Dict[str, Any]) -> np.ndarray:
    n = len(data["close"])
    signals = np.zeros(n, dtype=int)
    close = data["close"]
    entry_type = strategy.get("entry_type", "crossover")

    for i in range(2, n):
        buy_conditions = []
        sell_conditions = []

        # RSI
        if "RSI" in indicators:
            rsi = indicators["RSI"]
            oversold = indicators.get("RSI_oversold", 30)
            overbought = indicators.get("RSI_overbought", 70)
            if not np.isnan(rsi[i]):
                buy_conditions.append(rsi[i] < oversold or (not np.isnan(rsi[i-1]) and rsi[i - 1] < oversold and rsi[i] > rsi[i - 1]))
                sell_conditions.append(rsi[i] > overbought or (not np.isnan(rsi[i-1]) and rsi[i - 1] > overbought and rsi[i] < rsi[i - 1]))

        # MACD crossover
        if "MACD_main" in indicators and "MACD_signal" in indicators:
            macd_m = indicators["MACD_main"]
            macd_s = indicators["MACD_signal"]
            if (not np.isnan(macd_m[i]) and not np.isnan(macd_s[i]) and
                not np.isnan(macd_m[i - 1]) and not np.isnan(macd_s[i - 1])):
                buy_conditions.append(macd_m[i - 1] < macd_s[i - 1] and macd_m[i] > macd_s[i])
                sell_conditions.append(macd_m[i - 1] > macd_s[i - 1] and macd_m[i] < macd_s[i])

        # Stochastic
        if "STOCH_K" in indicators and "STOCH_D" in indicators:
            k = indicators["STOCH_K"]
            d = indicators["STOCH_D"]
            ob = indicators.get("STOCH_overbought", 80)
            os_val = indicators.get("STOCH_oversold", 20)
            if (not np.isnan(k[i]) and not np.isnan(d[i]) and
                not np.isnan(k[i - 1]) and not np.isnan(d[i - 1])):
                stoch_cross_up = k[i - 1] < d[i - 1] and k[i] > d[i]
                stoch_cross_down = k[i - 1] > d[i - 1] and k[i] < d[i]
                stoch_oversold = k[i] < os_val or k[i - 1] < os_val
                stoch_overbought = k[i] > ob or k[i - 1] > ob
                buy_conditions.append(stoch_cross_up and stoch_oversold)
                sell_conditions.append(stoch_cross_down and stoch_overbought)

        # EMA / SMA
        ema_keys = [k for k in indicators if isinstance(k, str) and k.startswith("EMA_")]
        sma_keys = [k for k in indicators if isinstance(k, str) and k.startswith("SMA_")]

        if ema_keys and sma_keys:
            ema = indicators[ema_keys[0]]
            sma = indicators[sma_keys[0]]
            if not np.isnan(ema[i]) and not np.isnan(sma[i]):
                buy_conditions.append(ema[i] > sma[i] and close[i] > ema[i])
                sell_conditions.append(ema[i] < sma[i] and close[i] < ema[i])
        elif ema_keys:
            ema = indicators[ema_keys[0]]
            if not np.isnan(ema[i]) and not np.isnan(ema[i - 1]):
                buy_conditions.append(close[i] > ema[i] and close[i - 1] <= ema[i - 1])
                sell_conditions.append(close[i] < ema[i] and close[i - 1] >= ema[i - 1])

        # Bollinger Bands
        if "BB_lower" in indicators and "BB_upper" in indicators:
            bb_lower = indicators["BB_lower"]
            bb_upper = indicators["BB_upper"]
            if not np.isnan(bb_lower[i]) and not np.isnan(bb_upper[i]):
                buy_conditions.append(close[i] <= bb_lower[i])
                sell_conditions.append(close[i] >= bb_upper[i])

        # ADX filtre
        if "ADX" in indicators:
            adx = indicators["ADX"]
            threshold = indicators.get("ADX_threshold", 25)
            if not np.isnan(adx[i]):
                if adx[i] < threshold:
                    buy_conditions = [False]
                    sell_conditions = [False]

        # Décision
        if entry_type == "crossover":
            if buy_conditions and any(buy_conditions):
                signals[i] = 1
            elif sell_conditions and any(sell_conditions):
                signals[i] = -1
        elif entry_type == "threshold":
            buy_count = sum(1 for c in buy_conditions if c)
            sell_count = sum(1 for c in sell_conditions if c)
            min_required = max(1, len(buy_conditions) // 2)
            if buy_count >= min_required:
                signals[i] = 1
            elif sell_count >= min_required:
                signals[i] = -1
        elif entry_type == "momentum":
            if buy_conditions and all(buy_conditions):
                signals[i] = 1
            elif sell_conditions and all(sell_conditions):
                signals[i] = -1
        else:
            if buy_conditions and any(buy_conditions):
                signals[i] = 1
            elif sell_conditions and any(sell_conditions):
                signals[i] = -1

    return signals


# ═══════════════════════════════════════════════════════
# SIMULATION DES TRADES AVEC SL/TP RÉELS
# ═══════════════════════════════════════════════════════

def simulate_trades(data: Dict[str, np.ndarray], signals: np.ndarray,
                     strategy: Dict[str, Any]) -> Dict[str, Any]:
    close = data["close"]
    high = data["high"]
    low = data["low"]
    n = len(close)
    point = data.get("point", 0.00001)

    rm = strategy.get("risk_management", {})
    sl_pips = rm.get("stop_loss", 50)
    tp_pips = rm.get("take_profit", 100)

    digits = data.get("digits", 5)
    if digits == 3 or digits == 5:
        pip_multiplier = 10.0
    else:
        pip_multiplier = 1.0

    sl_price = sl_pips * pip_multiplier * point
    tp_price = tp_pips * pip_multiplier * point

    in_trade = False
    trade_direction = 0
    entry_price = 0.0
    trade_sl = 0.0
    trade_tp = 0.0

    trade_results = []
    equity_curve = [0.0]
    cumulative_pips = 0.0
    max_equity = 0.0
    max_drawdown_pips = 0.0

    for i in range(1, n):
        if in_trade:
            if trade_direction == 1:
                if low[i] <= trade_sl:
                    pips = -sl_pips
                    trade_results.append(pips)
                    cumulative_pips += pips
                    in_trade = False
                elif high[i] >= trade_tp:
                    pips = tp_pips
                    trade_results.append(pips)
                    cumulative_pips += pips
                    in_trade = False
            elif trade_direction == -1:
                if high[i] >= trade_sl:
                    pips = -sl_pips
                    trade_results.append(pips)
                    cumulative_pips += pips
                    in_trade = False
                elif low[i] <= trade_tp:
                    pips = tp_pips
                    trade_results.append(pips)
                    cumulative_pips += pips
                    in_trade = False

            equity_curve.append(cumulative_pips)
            max_equity = max(max_equity, cumulative_pips)
            dd = max_equity - cumulative_pips
            max_drawdown_pips = max(max_drawdown_pips, dd)

            if not in_trade:
                continue
            else:
                continue

        if signals[i] == 1:
            in_trade = True
            trade_direction = 1
            entry_price = close[i]
            trade_sl = entry_price - sl_price
            trade_tp = entry_price + tp_price
        elif signals[i] == -1:
            in_trade = True
            trade_direction = -1
            entry_price = close[i]
            trade_sl = entry_price + sl_price
            trade_tp = entry_price - tp_price

        equity_curve.append(cumulative_pips)
        max_equity = max(max_equity, cumulative_pips)
        dd = max_equity - cumulative_pips
        max_drawdown_pips = max(max_drawdown_pips, dd)

    # Fermer trade en cours
    if in_trade:
        if trade_direction == 1:
            pips = (close[-1] - entry_price) / (pip_multiplier * point)
        else:
            pips = (entry_price - close[-1]) / (pip_multiplier * point)
        trade_results.append(round(pips, 1))
        cumulative_pips += pips

    # Statistiques
    total_trades = len(trade_results)
    if total_trades == 0:
        return {
            "profit": 0, "trades": 0, "winrate": 0, "drawdown": 0,
            "sharpe_ratio": 0, "profit_factor": 0, "trade_results": [],
            "equity_curve": [0], "date_start": "", "date_end": "",
            "data_source": "mt5_real", "bars_used": n
        }

    wins = sum(1 for t in trade_results if t > 0)
    winrate = wins / total_trades

    gross_profit = sum(t for t in trade_results if t > 0)
    gross_loss = abs(sum(t for t in trade_results if t < 0))
    profit_factor = gross_profit / max(gross_loss, 0.01)

    if len(trade_results) > 1:
        avg = np.mean(trade_results)
        std = np.std(trade_results)
        sharpe = (avg / std) * np.sqrt(252) / 10 if std > 0 else 0
    else:
        sharpe = 0

    try:
        date_start = datetime.fromtimestamp(int(data["time"][0])).strftime("%Y-%m-%d")
        date_end = datetime.fromtimestamp(int(data["time"][-1])).strftime("%Y-%m-%d")
    except Exception:
        date_start = "N/A"
        date_end = "N/A"

    return {
        "profit": round(cumulative_pips, 2),
        "trades": total_trades,
        "winrate": round(winrate, 4),
        "drawdown": round(max_drawdown_pips, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "trade_results": trade_results,
        "equity_curve": equity_curve,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(-gross_loss, 2),
        "date_start": date_start,
        "date_end": date_end,
        "bars_used": n,
        "data_source": "mt5_real",
        "symbol": data.get("symbol", "?"),
    }


# ═══════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════

def run_real_backtest(strategy: Dict[str, Any],
                       start: datetime = None,
                       end: datetime = None,
                       bars: int = 10000) -> Optional[Dict[str, Any]]:
    """
    Lance un VRAI backtest avec données MT5 réelles.
    Retourne None si MT5 non disponible ou pas de données.
    """
    symbol = strategy.get("symbol", "EURUSD")
    timeframe = strategy.get("timeframe", "H1")

    logger.info(f"🔬 Backtest RÉEL: {symbol} {timeframe}")

    # 1. Récupérer les données
    data = fetch_mt5_data(symbol, timeframe, bars=bars, start_date=start, end_date=end)
    if data is None:
        return None

    logger.info(f"   📊 {len(data['close'])} barres chargées")

    # 2. Calculer les indicateurs
    computed_indicators = compute_indicators(data, strategy)
    logger.info(f"   📈 {len(computed_indicators)} indicateurs calculés")

    # 3. Générer les signaux
    signals = generate_signals(data, computed_indicators, strategy)
    buy_signals = int(np.sum(signals == 1))
    sell_signals = int(np.sum(signals == -1))
    logger.info(f"   🎯 Signaux: {buy_signals} BUY, {sell_signals} SELL")

    if buy_signals + sell_signals == 0:
        logger.warning(f"   ⚠️ Aucun signal généré pour cette stratégie")
        return {
            "profit": 0, "trades": 0, "winrate": 0, "drawdown": 0,
            "sharpe_ratio": 0, "profit_factor": 0, "trade_results": [],
            "equity_curve": [0],
            "date_start": datetime.fromtimestamp(int(data["time"][0])).strftime("%Y-%m-%d"),
            "date_end": datetime.fromtimestamp(int(data["time"][-1])).strftime("%Y-%m-%d"),
            "data_source": "mt5_real", "bars_used": len(data["close"]),
            "symbol": data.get("symbol", symbol),
        }

    # 4. Simuler les trades
    result = simulate_trades(data, signals, strategy)
    logger.info(f"   💰 Résultat: {result['profit']:.1f} pips | {result['trades']} trades | WR: {result['winrate']:.1%}")

    return result