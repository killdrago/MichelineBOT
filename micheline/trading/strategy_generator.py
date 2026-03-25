# micheline/trading/strategy_generator.py
"""
Générateur de stratégies de trading.
"""

import random
import copy
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional


AVAILABLE_INDICATORS = [
    {"name": "SMA", "params": {"period": (5, 200)}},
    {"name": "EMA", "params": {"period": (5, 200)}},
    {"name": "DEMA", "params": {"period": (5, 200)}},
    {"name": "TEMA", "params": {"period": (5, 200)}},
    {"name": "RSI", "params": {"period": (5, 50)}},
    {"name": "Stochastic", "params": {"k_period": (5, 21), "d_period": (3, 14)}},
    {"name": "CCI", "params": {"period": (10, 50)}},
    {"name": "Williams_R", "params": {"period": (10, 30)}},
    {"name": "MFI", "params": {"period": (10, 30)}},
    {"name": "BollingerBands", "params": {"period": (10, 50), "deviation": (1.0, 3.0)}},
    {"name": "ATR", "params": {"period": (5, 30)}},
    {"name": "Keltner", "params": {"period": (10, 40), "multiplier": (1.0, 3.0)}},
    {"name": "OBV", "params": {}},
    {"name": "VWAP", "params": {}},
    {"name": "ADX", "params": {"period": (10, 30)}},
    {"name": "MACD", "params": {"fast": (8, 21), "slow": (21, 55), "signal": (5, 13)}},
    {"name": "Ichimoku", "params": {"tenkan": (7, 13), "kijun": (22, 30), "senkou": (44, 60)}},
]

AVAILABLE_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

AVAILABLE_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP",
    "EURCAD", "EURCHF", "GBPCAD", "GBPCHF",
    "CADJPY", "CHFJPY", "CADCHF",
    "XAUUSD", "XAGUSD",
    "Usa500", "UsaInd", "UsaTec",
    "Ger40", "UK100", "Fra40", "Jp225",
]

ENTRY_LOGIC_TEMPLATES = [
    "crossover", "threshold_above", "threshold_below",
    "divergence", "breakout", "mean_reversion", "multi_condition",
]

EXIT_LOGIC_TEMPLATES = [
    "fixed_tp_sl", "trailing_stop", "indicator_exit",
    "time_based", "atr_based", "opposite_signal",
]


def _safe_randint(low: int, high: int) -> int:
    """randint sécurisé qui gère low >= high."""
    low, high = int(low), int(high)
    if low >= high:
        return low
    return random.randint(low, high)


def _safe_uniform(low: float, high: float) -> float:
    """uniform sécurisé qui gère low >= high."""
    if low >= high:
        return low
    return random.uniform(low, high)


class StrategyGenerator:

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_indicators: int = 5,
        min_indicators: int = 1,
    ):
        self.symbols = symbols or AVAILABLE_SYMBOLS
        self.timeframes = timeframes or AVAILABLE_TIMEFRAMES
        self.max_indicators = max_indicators
        self.min_indicators = min_indicators
        self.generation_count = 0

    def generate_strategy(self) -> Dict[str, Any]:
        self.generation_count += 1
        strat_id = str(uuid.uuid4())[:8]

        symbol = random.choice(self.symbols)
        timeframe = random.choice(self.timeframes)

        nb_indicators = _safe_randint(self.min_indicators, self.max_indicators)
        indicators = self._generate_indicators(nb_indicators)
        entry_logic = self._generate_entry_logic(indicators)
        exit_logic = self._generate_exit_logic()
        risk_management = self._generate_risk_management()

        return {
            "id": f"strat_{strat_id}",
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators,
            "entry_logic": entry_logic,
            "exit_logic": exit_logic,
            "risk_management": risk_management,
            "metadata": {
                "generation": self.generation_count,
                "created_at": datetime.now().isoformat(),
                "parent_id": None,
                "mutation_type": None,
                "version": 1,
            },
        }

    def mutate_strategy(self, strategy: Dict[str, Any], mutation_strength: float = 0.3) -> Dict[str, Any]:
        mutated = copy.deepcopy(strategy)
        self.generation_count += 1

        parent_id = mutated["id"]
        mutated["id"] = f"strat_{str(uuid.uuid4())[:8]}"
        mutated["metadata"]["parent_id"] = parent_id
        mutated["metadata"]["generation"] = self.generation_count
        mutated["metadata"]["created_at"] = datetime.now().isoformat()
        mutated["metadata"]["version"] = mutated["metadata"].get("version", 1) + 1

        mutations_applied = []

        if random.random() < mutation_strength:
            mutated["indicators"] = self._mutate_indicators(mutated["indicators"], mutation_strength)
            mutations_applied.append("indicators")

        if random.random() < mutation_strength:
            mutated["risk_management"] = self._mutate_risk(mutated["risk_management"], mutation_strength)
            mutations_applied.append("risk_management")

        if random.random() < mutation_strength * 0.3:
            mutated["timeframe"] = random.choice(self.timeframes)
            mutations_applied.append("timeframe")

        if random.random() < mutation_strength * 0.5:
            mutated["exit_logic"] = self._generate_exit_logic()
            mutations_applied.append("exit_logic")

        mutated["metadata"]["mutation_type"] = mutations_applied or ["none"]
        return mutated

    def crossover(self, strategy_a: Dict[str, Any], strategy_b: Dict[str, Any]) -> Dict[str, Any]:
        self.generation_count += 1
        return {
            "id": f"strat_{str(uuid.uuid4())[:8]}",
            "symbol": random.choice([strategy_a["symbol"], strategy_b["symbol"]]),
            "timeframe": random.choice([strategy_a["timeframe"], strategy_b["timeframe"]]),
            "indicators": self._crossover_indicators(
                strategy_a.get("indicators", []),
                strategy_b.get("indicators", []),
            ),
            "entry_logic": random.choice([strategy_a["entry_logic"], strategy_b["entry_logic"]]),
            "exit_logic": random.choice([strategy_a["exit_logic"], strategy_b["exit_logic"]]),
            "risk_management": self._crossover_risk(
                strategy_a["risk_management"], strategy_b["risk_management"]
            ),
            "metadata": {
                "generation": self.generation_count,
                "created_at": datetime.now().isoformat(),
                "parent_id": f"{strategy_a['id']}+{strategy_b['id']}",
                "mutation_type": ["crossover"],
                "version": 1,
            },
        }

    def _generate_indicators(self, count: int) -> List[Dict[str, Any]]:
        count = max(1, min(count, len(AVAILABLE_INDICATORS)))
        selected = random.sample(AVAILABLE_INDICATORS, count)
        indicators = []

        for ind_template in selected:
            indicator = {"name": ind_template["name"], "params": {}}
            for param_name, param_range in ind_template["params"].items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    low, high = param_range
                    if isinstance(low, float):
                        indicator["params"][param_name] = round(_safe_uniform(low, high), 2)
                    else:
                        indicator["params"][param_name] = _safe_randint(low, high)
            indicators.append(indicator)

        return indicators

    def _generate_entry_logic(self, indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        logic_type = random.choice(ENTRY_LOGIC_TEMPLATES)
        entry = {
            "type": logic_type,
            "direction": random.choice(["long", "short", "both"]),
            "conditions": [],
        }

        if logic_type == "crossover" and len(indicators) >= 2:
            ind_pair = random.sample(indicators, 2)
            entry["conditions"].append({
                "type": "crossover",
                "fast_indicator": ind_pair[0]["name"],
                "slow_indicator": ind_pair[1]["name"],
            })
        elif logic_type in ("threshold_above", "threshold_below") and indicators:
            ind = random.choice(indicators)
            entry["conditions"].append({
                "type": logic_type,
                "indicator": ind["name"],
                "threshold": round(_safe_uniform(20, 80), 1),
            })
        elif logic_type == "breakout":
            entry["conditions"].append({
                "type": "breakout",
                "lookback_period": _safe_randint(10, 100),
                "confirmation_candles": _safe_randint(1, 3),
            })
        elif logic_type == "mean_reversion" and indicators:
            ind = random.choice(indicators)
            entry["conditions"].append({
                "type": "mean_reversion",
                "indicator": ind["name"],
                "deviation_threshold": round(_safe_uniform(1.5, 3.0), 2),
            })
        elif logic_type == "multi_condition" and indicators:
            nb = min(3, len(indicators))
            for ind in random.sample(indicators, nb):
                entry["conditions"].append({
                    "type": random.choice(["threshold_above", "threshold_below", "trend_filter"]),
                    "indicator": ind["name"],
                    "threshold": round(_safe_uniform(20, 80), 1),
                })
        elif logic_type == "divergence" and indicators:
            ind = random.choice(indicators)
            entry["conditions"].append({
                "type": "divergence",
                "indicator": ind["name"],
                "lookback": _safe_randint(5, 30),
            })

        return entry

    def _generate_exit_logic(self) -> Dict[str, Any]:
        logic_type = random.choice(EXIT_LOGIC_TEMPLATES)
        exit_logic = {"type": logic_type, "params": {}}

        if logic_type == "fixed_tp_sl":
            exit_logic["params"] = {
                "take_profit_pips": _safe_randint(10, 200),
                "stop_loss_pips": _safe_randint(10, 100),
            }
        elif logic_type == "trailing_stop":
            exit_logic["params"] = {
                "trail_distance_pips": _safe_randint(10, 80),
                "trail_step_pips": _safe_randint(1, 10),
                "initial_sl_pips": _safe_randint(20, 100),
            }
        elif logic_type == "atr_based":
            exit_logic["params"] = {
                "atr_period": _safe_randint(10, 25),
                "tp_atr_multiplier": round(_safe_uniform(1.5, 4.0), 2),
                "sl_atr_multiplier": round(_safe_uniform(0.5, 2.5), 2),
            }
        elif logic_type == "time_based":
            exit_logic["params"] = {
                "max_bars_in_trade": _safe_randint(5, 100),
                "fallback_sl_pips": _safe_randint(20, 80),
            }
        elif logic_type == "indicator_exit":
            exit_logic["params"] = {
                "exit_indicator": random.choice(["RSI", "CCI", "Stochastic"]),
                "exit_threshold_long": round(_safe_uniform(65, 85), 1),
                "exit_threshold_short": round(_safe_uniform(15, 35), 1),
            }
        elif logic_type == "opposite_signal":
            exit_logic["params"] = {
                "use_opposite_entry": True,
                "fallback_sl_pips": _safe_randint(30, 100),
            }

        return exit_logic

    def _generate_risk_management(self) -> Dict[str, Any]:
        return {
            "lot_size": round(_safe_uniform(0.01, 0.5), 2),
            "max_risk_percent": round(_safe_uniform(0.5, 3.0), 2),
            "max_open_trades": _safe_randint(1, 5),
            "max_daily_loss_percent": round(_safe_uniform(2.0, 10.0), 2),
            "position_sizing": random.choice(["fixed", "percent_equity", "kelly"]),
        }

    def _mutate_indicators(self, indicators: List[Dict[str, Any]], strength: float) -> List[Dict[str, Any]]:
        mutated = copy.deepcopy(indicators)

        for ind in mutated:
            for param_name, param_value in ind["params"].items():
                if random.random() < strength:
                    if isinstance(param_value, int):
                        delta = max(1, int(param_value * strength * 0.5))
                        new_val = param_value + _safe_randint(-delta, delta)
                        ind["params"][param_name] = max(1, new_val)
                    elif isinstance(param_value, float):
                        delta = max(0.01, param_value * strength * 0.5)
                        new_val = param_value + _safe_uniform(-delta, delta)
                        ind["params"][param_name] = round(max(0.01, new_val), 2)

        if random.random() < strength * 0.3 and len(mutated) < self.max_indicators:
            try:
                mutated.extend(self._generate_indicators(1))
            except Exception:
                pass

        if random.random() < strength * 0.2 and len(mutated) > self.min_indicators:
            mutated.pop(random.randint(0, len(mutated) - 1))

        return mutated

    def _mutate_risk(self, risk: Dict[str, Any], strength: float) -> Dict[str, Any]:
        mutated = copy.deepcopy(risk)

        if random.random() < strength:
            mutated["lot_size"] = round(max(0.01, mutated["lot_size"] + _safe_uniform(-0.05, 0.05)), 2)
        if random.random() < strength:
            mutated["max_risk_percent"] = round(max(0.1, min(5.0, mutated["max_risk_percent"] + _safe_uniform(-0.5, 0.5))), 2)
        if random.random() < strength:
            mutated["max_open_trades"] = max(1, mutated["max_open_trades"] + _safe_randint(-1, 1))

        return mutated

    def _crossover_indicators(self, indicators_a: List[Dict], indicators_b: List[Dict]) -> List[Dict]:
        all_indicators = indicators_a + indicators_b
        if not all_indicators:
            return self._generate_indicators(2)

        max_pick = min(self.max_indicators, len(all_indicators))
        min_pick = min(self.min_indicators, max_pick)
        nb = _safe_randint(min_pick, max_pick)
        nb = min(nb, len(all_indicators))
        if nb <= 0:
            return self._generate_indicators(1)
        return random.sample(all_indicators, nb)

    def _crossover_risk(self, risk_a: Dict[str, Any], risk_b: Dict[str, Any]) -> Dict[str, Any]:
        w = random.random()
        return {
            "lot_size": round(risk_a["lot_size"] * w + risk_b["lot_size"] * (1 - w), 2),
            "max_risk_percent": round(risk_a["max_risk_percent"] * w + risk_b["max_risk_percent"] * (1 - w), 2),
            "max_open_trades": max(1, int(risk_a["max_open_trades"] * w + risk_b["max_open_trades"] * (1 - w))),
            "max_daily_loss_percent": round(
                risk_a.get("max_daily_loss_percent", 5.0) * w + risk_b.get("max_daily_loss_percent", 5.0) * (1 - w), 2
            ),
            "position_sizing": random.choice([
                risk_a.get("position_sizing", "fixed"),
                risk_b.get("position_sizing", "fixed"),
            ]),
        }