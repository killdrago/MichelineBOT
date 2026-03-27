"""
micheline/trading/strategies/strategy_templates.py

Toutes les stratégies de trading nommées avec leurs variantes.
Chaque stratégie définit ses indicateurs, règles d'entrée/sortie et paramètres.
"""

import random
import time
import copy
from typing import Dict, Any, List, Optional

# ═══════════════════════════════════════════════════════
# CATALOGUE DE STRATÉGIES
# ═══════════════════════════════════════════════════════

def get_all_strategy_families() -> List[str]:
    """Retourne la liste de toutes les familles de stratégies disponibles."""
    return [
        "ichimoku",
        "ict_smc",
        "volume_profile",
        "crossover_ma",
        "multi_ma_ribbon",
        "rsi_divergence",
        "bollinger_squeeze",
        "macd_histogram",
        "supertrend",
        "donchian_breakout",
        "keltner_breakout",
        "stochastic_rsi",
        "pivot_bounce",
        "fibonacci_retracement",
        "cci_momentum",
        "williams_reversal",
        "parabolic_sar_trend",
        "atr_volatility",
        "obv_volume",
        "vwap_mean_reversion",
        "triple_screen",
        "combined_momentum",
        "structure_breakout",
        "hybrid_ichimoku_smc",
        "multi_indicator_fusion",
    ]


def generate_strategy_from_template(family: str, symbol: str = "EURUSD",
                                    timeframe: Optional[str] = None) -> dict:
    """
    Génère une stratégie à partir d'un template de famille.
    Les paramètres sont randomisés dans des plages raisonnables.
    """
    generators = {
        "ichimoku": _gen_ichimoku,
        "ict_smc": _gen_ict_smc,
        "volume_profile": _gen_volume_profile,
        "crossover_ma": _gen_crossover_ma,
        "multi_ma_ribbon": _gen_multi_ma_ribbon,
        "rsi_divergence": _gen_rsi_divergence,
        "bollinger_squeeze": _gen_bollinger_squeeze,
        "macd_histogram": _gen_macd_histogram,
        "supertrend": _gen_supertrend,
        "donchian_breakout": _gen_donchian_breakout,
        "keltner_breakout": _gen_keltner_breakout,
        "stochastic_rsi": _gen_stochastic_rsi,
        "pivot_bounce": _gen_pivot_bounce,
        "fibonacci_retracement": _gen_fibonacci_retracement,
        "cci_momentum": _gen_cci_momentum,
        "williams_reversal": _gen_williams_reversal,
        "parabolic_sar_trend": _gen_parabolic_sar_trend,
        "atr_volatility": _gen_atr_volatility,
        "obv_volume": _gen_obv_volume,
        "vwap_mean_reversion": _gen_vwap_mean_reversion,
        "triple_screen": _gen_triple_screen,
        "combined_momentum": _gen_combined_momentum,
        "structure_breakout": _gen_structure_breakout,
        "hybrid_ichimoku_smc": _gen_hybrid_ichimoku_smc,
        "multi_indicator_fusion": _gen_multi_indicator_fusion,
    }

    gen_func = generators.get(family, _gen_crossover_ma)
    strategy = gen_func(symbol, timeframe)
    strategy["family"] = family
    return strategy


def _make_id(family: str) -> str:
    return f"{family}_{int(time.time())}_{random.randint(1000, 9999)}"


def _random_tf(timeframe: Optional[str] = None) -> str:
    if timeframe:
        return timeframe
    return random.choice(["M5", "M15", "M30", "H1", "H4", "D1"])


def _random_sl_tp(tf: str) -> dict:
    """Génère SL/TP adaptés au timeframe avec RR variable."""
    sl_ranges = {
        "M1": (5, 15), "M5": (8, 25), "M15": (10, 35), "M30": (15, 50),
        "H1": (20, 70), "H4": (30, 120), "D1": (50, 200)
    }
    sl_min, sl_max = sl_ranges.get(tf, (20, 70))
    sl = random.randint(sl_min, sl_max)

    # RR variable : de 1.0 à 5.0
    rr = round(random.uniform(1.0, 5.0), 1)
    tp = int(sl * rr)

    return {
        "stop_loss": sl,
        "take_profit": tp,
        "risk_reward_ratio": rr,
        "risk_per_trade": round(random.uniform(0.5, 2.0), 1)
    }


# ═══════════════════════════════════════════════════════
# 🏯 ICHIMOKU
# ═══════════════════════════════════════════════════════

def _gen_ichimoku(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)
    mult = {"M5": 0.5, "M15": 0.7, "M30": 1.0, "H1": 1.0, "H4": 1.0, "D1": 1.0}.get(tf, 1.0)

    tenkan = random.choice([7, 9, 12, 15])
    kijun = random.choice([22, 26, 30, 35])
    senkou_b = random.choice([44, 52, 60, 65])

    # Variantes d'Ichimoku
    variant = random.choice(["kumo_breakout", "tk_cross", "chikou_cross", "kumo_twist", "full_ichimoku"])

    return {
        "id": _make_id("ichimoku"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "ichimoku",
        "variant": variant,
        "indicators": [
            {"type": "ICHIMOKU", "params": {
                "tenkan_period": tenkan,
                "kijun_period": kijun,
                "senkou_b_period": senkou_b,
                "displacement": 26
            }},
            {"type": "ADX", "params": {"period": 14, "threshold": random.randint(20, 30)}},
        ],
        "entry_type": "ichimoku_" + variant,
        "exit_type": random.choice(["ichimoku_exit", "opposite_signal", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# 💰 ICT / SMC
# ═══════════════════════════════════════════════════════

def _gen_ict_smc(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    variant = random.choice([
        "order_block_bounce",
        "fvg_fill",
        "bos_entry",
        "choch_reversal",
        "liquidity_grab",
        "smc_full",
    ])

    lookback = random.randint(5, 15)

    indicators = [
        {"type": "ICT_STRUCTURE", "params": {"lookback": lookback}},
        {"type": "ORDER_BLOCKS", "params": {"lookback": random.randint(8, 20)}},
        {"type": "FVG", "params": {}},
    ]

    # Ajouter un filtre optionnel
    if random.random() > 0.3:
        indicators.append(
            {"type": "EMA", "params": {"period": random.choice([50, 100, 200])}}
        )

    return {
        "id": _make_id("ict_smc"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "ict_smc",
        "variant": variant,
        "indicators": indicators,
        "entry_type": "ict_" + variant,
        "exit_type": random.choice(["opposite_signal", "structure_exit", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# 📊 VOLUME PROFILE
# ═══════════════════════════════════════════════════════

def _gen_volume_profile(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    variant = random.choice(["poc_bounce", "vah_val_breakout", "value_area_revert"])
    lookback = random.choice([50, 100, 150, 200])

    indicators = [
        {"type": "VOLUME_PROFILE", "params": {"lookback": lookback, "num_levels": 50}},
        {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
    ]

    if random.random() > 0.4:
        indicators.append({"type": "VWAP", "params": {"period": random.choice([20, 50])}})

    return {
        "id": _make_id("vol_profile"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "volume_profile",
        "variant": variant,
        "indicators": indicators,
        "entry_type": "volume_" + variant,
        "exit_type": random.choice(["opposite_signal", "poc_exit", "time_based"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# 📈 CROSSOVER MA (toutes variantes)
# ═══════════════════════════════════════════════════════

def _gen_crossover_ma(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    ma_types = ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA"]
    fast_type = random.choice(ma_types)
    slow_type = random.choice(ma_types)

    mult = {"M5": 0.7, "M15": 0.85, "M30": 1.0, "H1": 1.2, "H4": 1.5, "D1": 2.0}.get(tf, 1.0)

    fast_period = max(5, int(random.randint(5, 25) * mult))
    slow_period = max(fast_period + 5, int(random.randint(30, 100) * mult))

    indicators = [
        {"type": fast_type, "params": {"period": fast_period}},
        {"type": slow_type, "params": {"period": slow_period}},
    ]

    # Filtre optionnel
    filter_type = random.choice(["ADX", "RSI", "ATR", None])
    if filter_type == "ADX":
        indicators.append({"type": "ADX", "params": {"period": 14, "threshold": random.randint(20, 30)}})
    elif filter_type == "RSI":
        indicators.append({"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}})
    elif filter_type == "ATR":
        indicators.append({"type": "ATR", "params": {"period": 14}})

    return {
        "id": _make_id("crossover"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "crossover_ma",
        "variant": f"{fast_type}_{fast_period}x{slow_type}_{slow_period}",
        "indicators": indicators,
        "entry_type": "crossover",
        "exit_type": random.choice(["opposite_signal", "trailing_stop", "time_based"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# 🌈 MULTI MA RIBBON
# ═══════════════════════════════════════════════════════

def _gen_multi_ma_ribbon(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    ma_type = random.choice(["EMA", "SMA", "HMA"])
    periods = sorted(random.sample(range(5, 200), random.randint(3, 6)))

    indicators = [
        {"type": ma_type, "params": {"period": p}} for p in periods
    ]

    return {
        "id": _make_id("ma_ribbon"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "multi_ma_ribbon",
        "variant": f"{ma_type}_ribbon_{len(periods)}",
        "indicators": indicators,
        "entry_type": "ribbon_alignment",
        "exit_type": random.choice(["ribbon_cross", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# RSI DIVERGENCE
# ═══════════════════════════════════════════════════════

def _gen_rsi_divergence(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("rsi_div"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "rsi_divergence",
        "variant": random.choice(["regular", "hidden", "double"]),
        "indicators": [
            {"type": "RSI", "params": {
                "period": random.choice([9, 14, 21]),
                "overbought": random.randint(65, 80),
                "oversold": random.randint(20, 35)
            }},
            {"type": "EMA", "params": {"period": random.choice([20, 50, 100])}},
        ],
        "entry_type": "rsi_divergence",
        "exit_type": random.choice(["opposite_signal", "rsi_exit", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# BOLLINGER SQUEEZE
# ═══════════════════════════════════════════════════════

def _gen_bollinger_squeeze(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("bb_squeeze"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "bollinger_squeeze",
        "variant": random.choice(["squeeze_breakout", "band_walk", "mean_reversion"]),
        "indicators": [
            {"type": "BB", "params": {
                "period": random.choice([15, 20, 25]),
                "deviation": round(random.uniform(1.5, 2.5), 1)
            }},
            {"type": "KELTNER", "params": {
                "ema_period": random.choice([15, 20, 25]),
                "atr_period": random.choice([10, 14, 20]),
                "multiplier": round(random.uniform(1.0, 2.0), 1)
            }},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "squeeze_breakout",
        "exit_type": random.choice(["opposite_signal", "band_exit", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# MACD HISTOGRAM
# ═══════════════════════════════════════════════════════

def _gen_macd_histogram(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("macd_hist"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "macd_histogram",
        "variant": random.choice(["zero_cross", "histogram_reversal", "divergence"]),
        "indicators": [
            {"type": "MACD", "params": {
                "fast": random.choice([8, 12, 15]),
                "slow": random.choice([21, 26, 30]),
                "signal": random.choice([5, 9, 12])
            }},
            {"type": "EMA", "params": {"period": random.choice([50, 100, 200])}},
        ],
        "entry_type": "macd_signal",
        "exit_type": random.choice(["opposite_signal", "macd_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# SUPERTREND
# ═══════════════════════════════════════════════════════

def _gen_supertrend(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("supertrend"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "supertrend",
        "variant": random.choice(["trend_follow", "double_supertrend", "with_rsi"]),
        "indicators": [
            {"type": "SUPERTREND", "params": {
                "period": random.choice([7, 10, 14]),
                "multiplier": round(random.uniform(2.0, 4.0), 1)
            }},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "supertrend_flip",
        "exit_type": random.choice(["supertrend_exit", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# DONCHIAN BREAKOUT
# ═══════════════════════════════════════════════════════

def _gen_donchian_breakout(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("donchian"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "donchian_breakout",
        "variant": random.choice(["turtle", "channel_breakout", "pullback"]),
        "indicators": [
            {"type": "DONCHIAN", "params": {"period": random.choice([10, 20, 30, 55])}},
            {"type": "ATR", "params": {"period": 14}},
        ],
        "entry_type": "donchian_breakout",
        "exit_type": random.choice(["donchian_exit", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# KELTNER BREAKOUT
# ═══════════════════════════════════════════════════════

def _gen_keltner_breakout(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("keltner"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "keltner_breakout",
        "variant": random.choice(["breakout", "mean_reversion"]),
        "indicators": [
            {"type": "KELTNER", "params": {
                "ema_period": random.choice([15, 20, 25]),
                "atr_period": random.choice([10, 14]),
                "multiplier": round(random.uniform(1.5, 3.0), 1)
            }},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "keltner_breakout",
        "exit_type": random.choice(["keltner_exit", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# STOCHASTIC RSI
# ═══════════════════════════════════════════════════════

def _gen_stochastic_rsi(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("stoch_rsi"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "stochastic_rsi",
        "variant": random.choice(["overbought_oversold", "cross", "divergence"]),
        "indicators": [
            {"type": "STOCH_RSI", "params": {
                "rsi_period": random.choice([9, 14, 21]),
                "stoch_period": random.choice([9, 14]),
                "k_smooth": random.choice([3, 5]),
                "overbought": random.randint(70, 85),
                "oversold": random.randint(15, 30)
            }},
            {"type": "EMA", "params": {"period": random.choice([20, 50, 100])}},
        ],
        "entry_type": "stoch_rsi_signal",
        "exit_type": random.choice(["opposite_signal", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# PIVOT BOUNCE
# ═══════════════════════════════════════════════════════

def _gen_pivot_bounce(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("pivot"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "pivot_bounce",
        "variant": random.choice(["pp_bounce", "sr_breakout", "sr_rejection"]),
        "indicators": [
            {"type": "PIVOTS", "params": {"period": random.choice([10, 20, 50])}},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "pivot_signal",
        "exit_type": random.choice(["opposite_signal", "pivot_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# FIBONACCI RETRACEMENT
# ═══════════════════════════════════════════════════════

def _gen_fibonacci_retracement(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("fibo"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "fibonacci_retracement",
        "variant": random.choice(["38.2_bounce", "61.8_bounce", "golden_zone"]),
        "indicators": [
            {"type": "FIBONACCI", "params": {"lookback": random.choice([30, 50, 100])}},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
            {"type": "EMA", "params": {"period": random.choice([20, 50])}},
        ],
        "entry_type": "fib_bounce",
        "exit_type": random.choice(["fib_extension", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# CCI MOMENTUM
# ═══════════════════════════════════════════════════════

def _gen_cci_momentum(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("cci"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "cci_momentum",
        "variant": random.choice(["zero_cross", "extreme_reversal", "trend_confirm"]),
        "indicators": [
            {"type": "CCI", "params": {
                "period": random.choice([14, 20, 30]),
                "overbought": random.randint(100, 200),
                "oversold": random.randint(-200, -100)
            }},
            {"type": "EMA", "params": {"period": random.choice([20, 50, 100])}},
        ],
        "entry_type": "cci_signal",
        "exit_type": random.choice(["opposite_signal", "cci_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# WILLIAMS %R REVERSAL
# ═══════════════════════════════════════════════════════

def _gen_williams_reversal(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("williams"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "williams_reversal",
        "variant": random.choice(["overbought_sell", "oversold_buy", "failure_swing"]),
        "indicators": [
            {"type": "WILLIAMS_R", "params": {
                "period": random.choice([10, 14, 21]),
                "overbought": random.randint(-20, -10),
                "oversold": random.randint(-90, -80)
            }},
            {"type": "EMA", "params": {"period": random.choice([20, 50])}},
        ],
        "entry_type": "williams_signal",
        "exit_type": random.choice(["opposite_signal", "williams_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# PARABOLIC SAR TREND
# ═══════════════════════════════════════════════════════

def _gen_parabolic_sar_trend(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("psar"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "parabolic_sar_trend",
        "variant": random.choice(["sar_flip", "sar_with_ema", "sar_adx"]),
        "indicators": [
            {"type": "PARABOLIC_SAR", "params": {
                "af_start": round(random.uniform(0.01, 0.03), 3),
                "af_step": round(random.uniform(0.01, 0.03), 3),
                "af_max": round(random.uniform(0.15, 0.25), 2)
            }},
            {"type": "ADX", "params": {"period": 14, "threshold": random.randint(20, 30)}},
        ],
        "entry_type": "sar_flip",
        "exit_type": random.choice(["sar_exit", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# ATR VOLATILITY
# ═══════════════════════════════════════════════════════

def _gen_atr_volatility(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("atr_vol"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "atr_volatility",
        "variant": random.choice(["volatility_breakout", "atr_channel"]),
        "indicators": [
            {"type": "ATR", "params": {"period": random.choice([10, 14, 20])}},
            {"type": "EMA", "params": {"period": random.choice([20, 50])}},
            {"type": "ADX", "params": {"period": 14, "threshold": random.randint(20, 30)}},
        ],
        "entry_type": "atr_breakout",
        "exit_type": random.choice(["atr_exit", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# OBV VOLUME
# ═══════════════════════════════════════════════════════

def _gen_obv_volume(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("obv"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "obv_volume",
        "variant": random.choice(["obv_trend", "obv_divergence", "obv_breakout"]),
        "indicators": [
            {"type": "OBV", "params": {}},
            {"type": "EMA", "params": {"period": random.choice([20, 50])}},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "obv_signal",
        "exit_type": random.choice(["opposite_signal", "obv_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# VWAP MEAN REVERSION
# ═══════════════════════════════════════════════════════

def _gen_vwap_mean_reversion(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("vwap"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "vwap_mean_reversion",
        "variant": random.choice(["vwap_bounce", "vwap_cross", "vwap_bands"]),
        "indicators": [
            {"type": "VWAP", "params": {"period": random.choice([20, 50, 100])}},
            {"type": "BB", "params": {"period": 20, "deviation": 2.0}},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "vwap_signal",
        "exit_type": random.choice(["vwap_exit", "opposite_signal"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# TRIPLE SCREEN (Elder)
# ═══════════════════════════════════════════════════════

def _gen_triple_screen(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("triple_screen"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "triple_screen",
        "variant": "elder_triple_screen",
        "indicators": [
            {"type": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"type": "RSI", "params": {"period": random.choice([9, 14]), "overbought": 70, "oversold": 30}},
            {"type": "STOCH", "params": {
                "k_period": random.choice([9, 14]),
                "d_period": 3, "slowing": 3,
                "overbought": 80, "oversold": 20
            }},
            {"type": "EMA", "params": {"period": random.choice([13, 21, 34])}},
        ],
        "entry_type": "triple_screen",
        "exit_type": random.choice(["opposite_signal", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# COMBINED MOMENTUM
# ═══════════════════════════════════════════════════════

def _gen_combined_momentum(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    indicators = [
        {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        {"type": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"type": "STOCH", "params": {"k_period": 14, "d_period": 3, "slowing": 3,
                                     "overbought": 80, "oversold": 20}},
        {"type": "CCI", "params": {"period": 20, "overbought": 100, "oversold": -100}},
        {"type": "MOMENTUM", "params": {"period": random.choice([10, 14, 20])}},
    ]

    return {
        "id": _make_id("combined_mom"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "combined_momentum",
        "variant": "multi_oscillator",
        "indicators": indicators,
        "entry_type": "momentum_consensus",
        "exit_type": random.choice(["opposite_signal", "momentum_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# STRUCTURE BREAKOUT (support/resistance)
# ═══════════════════════════════════════════════════════

def _gen_structure_breakout(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("struct_break"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "structure_breakout",
        "variant": random.choice(["sr_breakout", "sr_bounce", "range_breakout"]),
        "indicators": [
            {"type": "LIQUIDITY", "params": {
                "lookback": random.choice([15, 20, 30]),
                "touch_count": random.choice([2, 3])
            }},
            {"type": "ATR", "params": {"period": 14}},
            {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        ],
        "entry_type": "structure_signal",
        "exit_type": random.choice(["opposite_signal", "structure_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# HYBRID ICHIMOKU + SMC
# ═══════════════════════════════════════════════════════

def _gen_hybrid_ichimoku_smc(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    return {
        "id": _make_id("ichi_smc"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "hybrid_ichimoku_smc",
        "variant": random.choice(["kumo_ob", "tk_fvg", "full_hybrid"]),
        "indicators": [
            {"type": "ICHIMOKU", "params": {
                "tenkan_period": 9, "kijun_period": 26,
                "senkou_b_period": 52, "displacement": 26
            }},
            {"type": "ICT_STRUCTURE", "params": {"lookback": 10}},
            {"type": "ORDER_BLOCKS", "params": {"lookback": 15}},
            {"type": "FVG", "params": {}},
        ],
        "entry_type": "hybrid_ichimoku_smc",
        "exit_type": random.choice(["opposite_signal", "ichimoku_exit"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# MULTI INDICATOR FUSION
# ═══════════════════════════════════════════════════════

def _gen_multi_indicator_fusion(symbol: str, timeframe: Optional[str] = None) -> dict:
    tf = _random_tf(timeframe)

    all_indicators = [
        {"type": "RSI", "params": {"period": 14, "overbought": 70, "oversold": 30}},
        {"type": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"type": "BB", "params": {"period": 20, "deviation": 2.0}},
        {"type": "SUPERTREND", "params": {"period": 10, "multiplier": 3.0}},
        {"type": "ICHIMOKU", "params": {"tenkan_period": 9, "kijun_period": 26,
                                        "senkou_b_period": 52, "displacement": 26}},
        {"type": "STOCH", "params": {"k_period": 14, "d_period": 3, "slowing": 3,
                                     "overbought": 80, "oversold": 20}},
        {"type": "ADX", "params": {"period": 14, "threshold": 25}},
        {"type": "CCI", "params": {"period": 20, "overbought": 100, "oversold": -100}},
        {"type": "EMA", "params": {"period": random.choice([20, 50, 100, 200])}},
        {"type": "PARABOLIC_SAR", "params": {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2}},
    ]

    num = random.randint(3, 6)
    selected = random.sample(all_indicators, min(num, len(all_indicators)))

    return {
        "id": _make_id("fusion"),
        "symbol": symbol,
        "timeframe": tf,
        "strategy_type": "multi_indicator_fusion",
        "variant": f"fusion_{num}_indicators",
        "indicators": selected,
        "entry_type": "fusion_consensus",
        "exit_type": random.choice(["opposite_signal", "trailing_stop"]),
        "risk_management": _random_sl_tp(tf),
    }


# ═══════════════════════════════════════════════════════
# MUTATION DE STRATÉGIE
# ═══════════════════════════════════════════════════════

def mutate_strategy(strategy: dict, strength: float = 0.2) -> dict:
    """Crée une mutation d'une stratégie existante."""
    mutated = copy.deepcopy(strategy)
    mutated["id"] = f"{strategy.get('family', 'mut')}_{int(time.time())}_{random.randint(1000, 9999)}"

    # Muter les paramètres des indicateurs
    for ind in mutated.get("indicators", []):
        if "params" in ind:
            for key, val in ind["params"].items():
                if isinstance(val, int) and val > 0:
                    delta = max(1, int(val * strength))
                    ind["params"][key] = max(1, val + random.randint(-delta, delta))
                elif isinstance(val, float) and val > 0:
                    delta = val * strength
                    ind["params"][key] = round(max(0.01, val + random.uniform(-delta, delta)), 3)

    # Muter le risk management
    rm = mutated.get("risk_management", {})
    if rm:
        sl = rm.get("stop_loss", 50)
        sl_delta = max(1, int(sl * strength))
        new_sl = max(5, sl + random.randint(-sl_delta, sl_delta))
        rm["stop_loss"] = new_sl

        # Muter le RR
        rr = rm.get("risk_reward_ratio", 2.0)
        new_rr = round(max(0.5, rr + random.uniform(-0.5, 0.5)), 1)
        rm["risk_reward_ratio"] = new_rr
        rm["take_profit"] = int(new_sl * new_rr)

    return mutated