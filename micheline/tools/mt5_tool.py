# micheline/tools/mt5_tool.py
"""
Outils d'interaction avec MetaTrader 5.
Phase 5 + Phase 6 (Trading Engine compatible).
Fallback automatique sur simulation si données indisponibles.
"""

import logging
import random
import hashlib
from typing import Dict, Any, Optional

logger = logging.getLogger("micheline.tools.mt5_tool")

# Essayer d'importer MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 non installé — mode simulation activé")

# Cache des symboles disponibles
_available_symbols = None
_mt5_connected = False


def _ensure_connected() -> bool:
    """S'assure que MT5 est connecté."""
    global _mt5_connected
    if not MT5_AVAILABLE:
        return False
    if not _mt5_connected:
        if mt5.initialize():
            _mt5_connected = True
        else:
            logger.warning(f"MT5 init échoué: {mt5.last_error()}")
            return False
    return True


def _get_available_symbols() -> list:
    """Récupère et cache la liste des symboles disponibles dans MT5."""
    global _available_symbols
    if _available_symbols is not None:
        return _available_symbols

    if not _ensure_connected():
        _available_symbols = []
        return _available_symbols

    try:
        symbols = mt5.symbols_get()
        if symbols:
            _available_symbols = [s.name for s in symbols if s.visible]
            logger.info(f"MT5: {len(_available_symbols)} symboles disponibles")
        else:
            _available_symbols = []
    except Exception as e:
        logger.warning(f"Erreur récupération symboles: {e}")
        _available_symbols = []

    return _available_symbols


def _get_mt5_timeframe(tf_str: str):
    """Convertit un string timeframe en constante MT5."""
    if not MT5_AVAILABLE:
        return None

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return tf_map.get(tf_str.upper(), mt5.TIMEFRAME_H1)


def initialize_mt5() -> Dict[str, Any]:
    """Initialise la connexion à MT5."""
    global _mt5_connected

    if not MT5_AVAILABLE:
        return {"success": False, "error": "MetaTrader5 non installé"}

    if not mt5.initialize():
        return {"success": False, "error": f"MT5 init failed: {mt5.last_error()}"}

    _mt5_connected = True
    info = mt5.terminal_info()
    return {
        "success": True,
        "terminal": info.name if info else "unknown",
        "connected": True,
    }


def shutdown_mt5() -> Dict[str, Any]:
    """Ferme la connexion MT5."""
    global _mt5_connected
    if MT5_AVAILABLE:
        mt5.shutdown()
        _mt5_connected = False
    return {"success": True}


def run_backtest(config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Exécute un backtest.
    Retourne les résultats avec les dates de la période testée.
    """
    if config is None:
        config = kwargs
    elif kwargs:
        config = {**config, **kwargs}

    symbol = config.get("symbol", "EURUSD")
    timeframe = config.get("timeframe", "H1")

    # Essayer avec les vraies données MT5
    if _ensure_connected():
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            for suffix in [".raw", ".pro", ".std", ".ecn", ""]:
                alt = symbol + suffix
                alt_info = mt5.symbol_info(alt)
                if alt_info is not None:
                    symbol = alt
                    symbol_info = alt_info
                    break

        if symbol_info is not None:
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)

            mt5_tf = _get_mt5_timeframe(timeframe)
            bars_count = config.get("bars_count", 5000)
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars_count)

            if rates is not None and len(rates) > 10:
                logger.debug(f"Backtest réel : {symbol} {timeframe} ({len(rates)} barres)")
                return _simulate_trades_on_data(rates, config, symbol)
            else:
                logger.debug(f"Pas de données pour {symbol} {timeframe}, fallback simulation")
        else:
            logger.debug(f"Symbole {symbol} non trouvé dans MT5, fallback simulation")

    return _run_simulated_backtest(config, symbol, timeframe)


def _simulate_trades_on_data(
    rates, config: Dict[str, Any], symbol: str
) -> Dict[str, Any]:
    """Simule des trades sur des données OHLCV réelles de MT5."""
    from datetime import datetime

    initial_deposit = 10000.0
    balance = initial_deposit
    max_balance = balance
    max_drawdown = 0.0
    trades_count = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    risk_mgmt = config.get("risk_management", {})
    lot_size = risk_mgmt.get("lot_size", 0.1)
    exit_logic = config.get("exit_logic", {})

    tp_pips = 50
    sl_pips = 30

    exit_type = exit_logic.get("type", "fixed_tp_sl")
    if exit_type == "fixed_tp_sl":
        tp_pips = exit_logic.get("params", {}).get("take_profit_pips", 50)
        sl_pips = exit_logic.get("params", {}).get("stop_loss_pips", 30)
    elif exit_type == "atr_based":
        tp_pips = 60
        sl_pips = 30
    elif exit_type == "trailing_stop":
        tp_pips = 80
        sl_pips = exit_logic.get("params", {}).get("initial_sl_pips", 40)

    if "JPY" in symbol:
        pip_value = 0.01
    elif "XAU" in symbol:
        pip_value = 0.1
    elif "XAG" in symbol:
        pip_value = 0.01
    elif symbol in ("Usa500", "UsaInd", "UsaTec", "Ger40", "UK100", "Fra40", "Jp225",
                     "US500", "US30", "GER40"):
        pip_value = 1.0
    else:
        pip_value = 0.0001

    trade_value_per_pip = lot_size * 100000 * pip_value

    if symbol in ("Usa500", "UsaInd", "UsaTec", "Ger40", "UK100", "Fra40", "Jp225",
                   "US500", "US30", "GER40"):
        trade_value_per_pip = lot_size * 10

    trade_interval = max(3, len(rates) // 200)

    config_str = str(config.get("id", "")) + symbol
    seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # ── Dates de la période ──
    try:
        date_start = datetime.fromtimestamp(int(rates[0]['time']))
        date_end = datetime.fromtimestamp(int(rates[-1]['time']))
    except Exception:
        date_start = None
        date_end = None

    for i in range(trade_interval, len(rates) - trade_interval, trade_interval):
        bar_open = rates[i]
        bar_close = rates[min(i + trade_interval, len(rates) - 1)]

        direction = 1 if rng.random() > 0.45 else -1
        price_move = (float(bar_close['close']) - float(bar_open['close'])) * direction
        pips_moved = price_move / pip_value if pip_value > 0 else 0

        if pips_moved >= tp_pips:
            pips_result = tp_pips
        elif pips_moved <= -sl_pips:
            pips_result = -sl_pips
        else:
            pips_result = pips_moved

        trade_pnl = pips_result * trade_value_per_pip
        balance += trade_pnl
        trades_count += 1

        if trade_pnl > 0:
            wins += 1
            gross_profit += trade_pnl
        else:
            gross_loss += trade_pnl

        max_balance = max(max_balance, balance)
        current_dd = ((max_balance - balance) / max_balance) * 100 if max_balance > 0 else 0
        max_drawdown = max(max_drawdown, current_dd)

        if balance <= initial_deposit * 0.1:
            break

    winrate = (wins / trades_count * 100) if trades_count > 0 else 0

    return {
        "profit": round(balance - initial_deposit, 2),
        "drawdown": round(max_drawdown, 2),
        "trades": trades_count,
        "winrate": round(winrate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "initial_deposit": initial_deposit,
        "final_balance": round(balance, 2),
        "symbol": symbol,
        "data_source": "mt5_real",
        "bars_used": len(rates),
        "date_start": date_start.strftime("%Y-%m-%d") if date_start else "N/A",
        "date_end": date_end.strftime("%Y-%m-%d") if date_end else "N/A",
    }


def _run_simulated_backtest(
    config: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    """Backtest simulé quand les données MT5 ne sont pas disponibles."""
    from datetime import datetime, timedelta

    config_str = str(sorted(str(config).encode()))
    seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    is_full_strategy = "indicators" in config and "entry_logic" in config
    quality = _estimate_strategy_quality(config, is_full_strategy, rng)

    initial_deposit = 10000.0
    base_trades = rng.randint(20, 200)
    trades = int(base_trades * quality["trade_frequency"])
    trades = max(5, trades)

    base_winrate = 45 + quality["quality_bonus"] * 15
    winrate = max(20, min(80, base_winrate + rng.gauss(0, 8)))

    winning_trades = int(trades * winrate / 100)
    losing_trades = trades - winning_trades

    avg_win = rng.uniform(10, 100) * quality["reward_factor"]
    avg_loss = rng.uniform(10, 80) * quality["risk_factor"]

    gross_profit = round(winning_trades * avg_win, 2)
    gross_loss = round(losing_trades * avg_loss, 2)
    net_profit = round(gross_profit - gross_loss, 2)

    max_dd = rng.uniform(5, 40) * quality["risk_factor"]
    drawdown = round(min(max_dd, 60), 2)

    # ── Dates simulées ──
    date_end = datetime.now()
    tf_days = {"M1": 30, "M5": 60, "M15": 120, "M30": 180,
               "H1": 365, "H4": 730, "D1": 1825}
    days_back = tf_days.get(timeframe, 365)
    date_start = date_end - timedelta(days=days_back)

    return {
        "profit": net_profit,
        "drawdown": drawdown,
        "trades": trades,
        "winrate": round(winrate, 2),
        "gross_profit": gross_profit,
        "gross_loss": -gross_loss,
        "initial_deposit": initial_deposit,
        "symbol": symbol,
        "data_source": "simulated",
        "date_start": date_start.strftime("%Y-%m-%d"),
        "date_end": date_end.strftime("%Y-%m-%d"),
    }

def _estimate_strategy_quality(
    config: Dict[str, Any], is_full_strategy: bool, rng
) -> Dict[str, float]:
    """Estime des facteurs de qualité à partir des paramètres."""
    quality_bonus = 0.0
    trade_frequency = 1.0
    reward_factor = 1.0
    risk_factor = 1.0

    if is_full_strategy:
        indicators = config.get("indicators", [])
        exit_logic = config.get("exit_logic", {})
        risk_mgmt = config.get("risk_management", {})

        if 2 <= len(indicators) <= 4:
            quality_bonus += 0.2
        elif len(indicators) > 4:
            quality_bonus -= 0.1

        exit_type = exit_logic.get("type", "")
        if exit_type in ("atr_based", "trailing_stop"):
            quality_bonus += 0.15
            reward_factor *= 1.2

        lot_size = risk_mgmt.get("lot_size", 0.1)
        if lot_size > 0.3:
            risk_factor *= 1.3
        elif lot_size < 0.05:
            reward_factor *= 0.7

        max_risk = risk_mgmt.get("max_risk_percent", 2.0)
        if max_risk > 3.0:
            risk_factor *= 1.2
        elif max_risk < 1.0:
            quality_bonus += 0.1

        tf = config.get("timeframe", "H1")
        tf_freq = {
            "M1": 3.0, "M5": 2.5, "M15": 2.0, "M30": 1.5,
            "H1": 1.0, "H4": 0.6, "D1": 0.3,
        }
        trade_frequency = tf_freq.get(tf, 1.0)

    quality_bonus += rng.gauss(0, 0.15)
    quality_bonus = max(-0.5, min(1.0, quality_bonus))

    return {
        "quality_bonus": quality_bonus,
        "trade_frequency": trade_frequency,
        "reward_factor": reward_factor,
        "risk_factor": risk_factor,
    }


def get_symbol_info(symbol: str = "EURUSD") -> Dict[str, Any]:
    """Récupère les infos d'un symbole MT5."""
    if not _ensure_connected():
        return {"symbol": symbol, "error": "MT5 non connecté", "simulated": True}

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"symbol": symbol, "error": "Symbole non trouvé"}

    return {
        "symbol": symbol,
        "bid": info.bid,
        "ask": info.ask,
        "spread": info.spread,
        "digits": info.digits,
        "trade_mode": info.trade_mode,
    }


def get_account_info() -> Dict[str, Any]:
    """Récupère les infos du compte MT5."""
    if not _ensure_connected():
        return {"error": "MT5 non connecté", "simulated": True}

    info = mt5.account_info()
    if info is None:
        return {"error": "Impossible de récupérer les infos compte"}

    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "profit": info.profit,
        "leverage": info.leverage,
    }