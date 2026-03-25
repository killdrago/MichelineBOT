"""
micheline/tools/trading_tools.py

Outils de trading pour l'agent Micheline.
Fonctionne AVEC ou SANS MT5 (mode simulation si MT5 indisponible).
"""

import logging
import random
import time
import copy
from datetime import datetime, timedelta
from typing import Dict, Any, List

logger = logging.getLogger("micheline.tools.trading")

# ═══════════════════════════════════
# Vérification MT5
# ═══════════════════════════════════
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 non installé — mode simulation activé")


def _is_mt5_connected():
    if not MT5_AVAILABLE:
        return False
    try:
        if not mt5.initialize():
            return False
        return mt5.account_info() is not None
    except Exception:
        return False


# ═══════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════

AVAILABLE_INDICATORS = [
    {"name": "SMA", "params": {"period": [5, 10, 20, 50, 100, 200]}},
    {"name": "EMA", "params": {"period": [5, 10, 20, 50, 100, 200]}},
    {"name": "RSI", "params": {"period": [7, 14, 21], "overbought": [70, 75, 80], "oversold": [20, 25, 30]}},
    {"name": "MACD", "params": {"fast": [8, 12, 16], "slow": [21, 26, 30], "signal": [5, 9, 12]}},
    {"name": "BB", "params": {"period": [14, 20, 25], "std_dev": [1.5, 2.0, 2.5]}},
    {"name": "ATR", "params": {"period": [7, 14, 21]}},
    {"name": "STOCH", "params": {"k_period": [5, 9, 14], "d_period": [3, 5], "overbought": [75, 80], "oversold": [20, 25]}},
]

AVAILABLE_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
AVAILABLE_ENTRY_LOGIC = ["crossover", "threshold", "reversal", "breakout", "mean_reversion"]
AVAILABLE_EXIT_LOGIC = ["opposite_signal", "fixed_tp_sl", "trailing_stop", "time_based", "indicator_exit"]


# ═══════════════════════════════════
# GÉNÉRATEUR
# ═══════════════════════════════════

def generate_random_strategy(symbol="EURUSD"):
    n_indicators = random.randint(2, 3)
    chosen = random.sample(AVAILABLE_INDICATORS, n_indicators)
    indicators = []
    for tpl in chosen:
        ind = {"name": tpl["name"], "params": {}}
        for pname, pvalues in tpl["params"].items():
            ind["params"][pname] = random.choice(pvalues)
        indicators.append(ind)

    sl = random.choice([15, 20, 25, 30, 40, 50, 60, 80])
    rr = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])

    return {
        "id": f"strat_{int(time.time())}_{random.randint(1000, 9999)}",
        "symbol": symbol,
        "timeframe": random.choice(AVAILABLE_TIMEFRAMES),
        "indicators": indicators,
        "entry_logic": random.choice(AVAILABLE_ENTRY_LOGIC),
        "exit_logic": random.choice(AVAILABLE_EXIT_LOGIC),
        "risk_management": {
            "sl_pips": sl, "tp_pips": int(sl * rr),
            "risk_reward_ratio": rr,
            "risk_per_trade_pct": random.choice([0.5, 1.0, 1.5, 2.0]),
            "max_open_trades": random.choice([1, 2, 3]),
        },
        "filters": {
            "min_spread": random.choice([0, 1, 2]),
            "trading_hours": random.choice(["all", "london", "new_york", "london_new_york_overlap"]),
        },
        "generated_at": datetime.now().isoformat(),
    }


def improve_strategy(strategy, score):
    improved = copy.deepcopy(strategy)
    mutation_strength = max(0.1, 1.0 - (score / 100.0))

    for ind in improved.get("indicators", []):
        for pname, pval in ind.get("params", {}).items():
            if isinstance(pval, (int, float)) and random.random() < mutation_strength:
                change = int(pval * random.uniform(-0.3, 0.3) * mutation_strength)
                ind["params"][pname] = max(1, pval + change)

    rm = improved.get("risk_management", {})
    if random.random() < mutation_strength * 0.5:
        rm["sl_pips"] = max(10, rm.get("sl_pips", 30) + random.randint(-10, 10))
        rm["tp_pips"] = max(10, rm.get("tp_pips", 60) + random.randint(-15, 15))
        if rm["sl_pips"] > 0:
            rm["risk_reward_ratio"] = round(rm["tp_pips"] / rm["sl_pips"], 2)

    if random.random() < mutation_strength * 0.2:
        improved["timeframe"] = random.choice(AVAILABLE_TIMEFRAMES)

    improved["id"] = f"{strategy['id']}_mut{random.randint(100, 999)}"
    return improved


# ═══════════════════════════════════
# SIMULATEUR BACKTEST
# ═══════════════════════════════════

def _simulate_backtest(strategy, start=None, end=None):
    strat_seed = hash(str(strategy.get("indicators", []))) % 2**32
    rng = random.Random(strat_seed)

    n_ind = len(strategy.get("indicators", []))
    rr = strategy.get("risk_management", {}).get("risk_reward_ratio", 1.5)
    sl = strategy.get("risk_management", {}).get("sl_pips", 30)
    tp = strategy.get("risk_management", {}).get("tp_pips", 45)

    base_wr = 0.40 + (n_ind * 0.03)
    wr_adj = max(-0.10, 0.05 - (rr - 1.5) * 0.05)
    winrate = min(0.65, max(0.30, base_wr + wr_adj + rng.uniform(-0.08, 0.08)))

    tf = strategy.get("timeframe", "H1")
    tpd = {"M1": 8, "M5": 5, "M15": 3, "M30": 2, "H1": 1.5, "H4": 0.5, "D1": 0.15}.get(tf, 1.0)

    n_days = (end - start).days if start and end else 180
    n_trades = max(10, int(n_days * tpd * rng.uniform(0.7, 1.3)))

    trade_results = []
    for _ in range(n_trades):
        if rng.random() < winrate:
            trade_results.append(round(tp * rng.uniform(0.6, 1.1), 2))
        else:
            trade_results.append(round(-sl * rng.uniform(0.7, 1.0), 2))

    total_profit = sum(trade_results)
    wins = [t for t in trade_results if t > 0]
    losses = [t for t in trade_results if t < 0]
    actual_wr = len(wins) / n_trades if n_trades > 0 else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    equity = [0]
    for t in trade_results:
        equity.append(equity[-1] + t)
    peak = equity[0]
    max_dd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    if len(trade_results) > 1:
        mean_r = sum(trade_results) / len(trade_results)
        var = sum((r - mean_r) ** 2 for r in trade_results) / (len(trade_results) - 1)
        std = var ** 0.5
        sharpe = (mean_r / std * (252 ** 0.5)) if std > 0 else 0
    else:
        sharpe = 0

    return {
        "profit": round(total_profit, 2),
        "drawdown": round(max_dd, 2),
        "trades": n_trades,
        "winrate": round(actual_wr, 4),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "trade_results": trade_results,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
        "max_drawdown_pips": round(max_dd, 2),
        "equity_curve": equity,
        "simulated": True,
        "symbol": strategy.get("symbol", "EURUSD"),
        "timeframe": strategy.get("timeframe", "H1"),
        "period_days": n_days,
    }


def run_backtest(strategy, start=None, end=None):
    return _simulate_backtest(strategy, start, end)


# ═══════════════════════════════════
# RECHERCHE DE STRATÉGIE
# ═══════════════════════════════════

def trading_search(params):
    try:
        symbols = params.get("symbols", ["EURUSD"])
        pop_size = min(params.get("population_size", 10), 50)
        max_gen = min(params.get("max_generations", 3), 10)

        data_end = datetime.now()
        data_start = data_end - timedelta(days=365)
        symbol = symbols[0] if symbols else "EURUSD"

        logger.info(f"=== Recherche: {symbol} | pop={pop_size} | gen={max_gen} ===")

        # Génération initiale
        population = []
        for i in range(pop_size):
            strat = generate_random_strategy(symbol)
            result = run_backtest(strat, data_start, data_end)

            try:
                from micheline.trading.metrics import evaluate_strategy
            except ImportError:
                try:
                    from trading.metrics import evaluate_strategy
                except ImportError:
                    evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50 + r.get("winrate", 0) * 30))

            score = evaluate_strategy(result)
            population.append({"strategy": strat, "result": result, "score": score, "generation": 0})
            logger.info(f"  Gen 0 | #{i+1}: score={score:.1f} | profit={result['profit']:.0f} | wr={result['winrate']:.1%}")

        # Évolution
        for gen in range(1, max_gen + 1):
            population.sort(key=lambda x: x["score"], reverse=True)
            survivors = population[:max(2, pop_size // 2)]
            new_pop = list(survivors)

            while len(new_pop) < pop_size:
                parent = random.choice(survivors)
                child = improve_strategy(parent["strategy"], parent["score"])
                child_result = run_backtest(child, data_start, data_end)
                child_score = evaluate_strategy(child_result)
                new_pop.append({"strategy": child, "result": child_result, "score": child_score, "generation": gen})

            population = new_pop
            best = max(population, key=lambda x: x["score"])
            logger.info(f"  Gen {gen} | Best: score={best['score']:.1f} | profit={best['result']['profit']:.0f}")

        # Résultat
        population.sort(key=lambda x: x["score"], reverse=True)
        best = population[0]
        top_3 = population[:3]

        # Anti-overfit (optionnel)
        anti_overfit_result = None
        try:
            try:
                from micheline.trading.anti_overfit import AntiOverfitValidator
            except ImportError:
                from trading.anti_overfit import AntiOverfitValidator

            validator = AntiOverfitValidator(
                backtest_runner=lambda config, s, e: run_backtest(config, s, e),
                n_folds=3
            )
            report = validator.quick_validate(
                train_metrics=best["result"],
                test_metrics=run_backtest(best["strategy"], data_end - timedelta(days=90), data_end),
                strategy_id=best["strategy"]["id"]
            )
            anti_overfit_result = {
                "verdict": report.verdict.value,
                "is_valid": report.is_valid,
                "degradation_ratio": report.degradation_ratio,
                "consistency_score": report.consistency_score
            }
        except Exception as e:
            logger.warning(f"Anti-overfit skipped: {e}")

        output = {
            "success": True,
            "symbol": symbol,
            "best_strategy": best["strategy"],
            "best_result": {k: v for k, v in best["result"].items() if k not in ("trade_results", "equity_curve")},
            "best_score": best["score"],
            "generations": max_gen,
            "total_evaluated": pop_size * (max_gen + 1),
            "top_3": [
                {"id": p["strategy"]["id"], "score": p["score"], "profit": p["result"]["profit"],
                 "winrate": p["result"]["winrate"], "drawdown": p["result"]["drawdown"],
                 "sharpe": p["result"]["sharpe_ratio"]}
                for p in top_3
            ],
            "anti_overfit": anti_overfit_result,
            "mt5_connected": MT5_AVAILABLE and _is_mt5_connected(),
            "mode": "mt5" if (MT5_AVAILABLE and _is_mt5_connected()) else "simulation"
        }

        output["formatted"] = format_strategy_summary(output)
        logger.info(f"=== Recherche terminée: score={best['score']:.1f} ===")
        return output

    except Exception as e:
        logger.error(f"Erreur trading_search: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


# ═══════════════════════════════════
# GÉNÉRATION SIMPLE (fallback)
# ═══════════════════════════════════

def trading_generate(params):
    try:
        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        strategy = generate_random_strategy(symbol)
        result = run_backtest(strategy)

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50 + r.get("winrate", 0) * 30))

        score = evaluate_strategy(result)

        output = {
            "success": True,
            "symbol": symbol,
            "best_strategy": strategy,
            "best_result": {k: v for k, v in result.items() if k not in ("trade_results", "equity_curve")},
            "best_score": score,
            "generations": 0,
            "total_evaluated": 1,
            "top_3": [{"id": strategy["id"], "score": score, "profit": result["profit"],
                       "winrate": result["winrate"], "drawdown": result["drawdown"],
                       "sharpe": result["sharpe_ratio"]}],
            "anti_overfit": None,
            "mode": "simulation",
            "mt5_connected": MT5_AVAILABLE and _is_mt5_connected()
        }
        output["formatted"] = format_strategy_summary(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_generate: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


# ═══════════════════════════════════
# FORMAT RÉSUMÉ
# ═══════════════════════════════════

def format_strategy_summary(search_result):
    if not search_result.get("success"):
        return f"❌ Recherche échouée: {search_result.get('error', 'Erreur inconnue')}"

    best = search_result.get("best_strategy", {})
    result = search_result.get("best_result", {})
    score = search_result.get("best_score", 0)

    indicators_str = ", ".join(
        f"{ind['name']}({', '.join(f'{k}={v}' for k, v in ind.get('params', {}).items())})"
        for ind in best.get("indicators", [])
    )
    rm = best.get("risk_management", {})

    summary = f"""
📊 **STRATÉGIE TROUVÉE — {best.get('symbol', '?')}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🆔 ID: {best.get('id', '?')}
⏱️ Timeframe: {best.get('timeframe', '?')}

📈 **Indicateurs:** {indicators_str}
🎯 **Entrée:** {best.get('entry_logic', '?')}
🚪 **Sortie:** {best.get('exit_logic', '?')}

💰 **Résultats (backtest):**
   • Profit: {result.get('profit', 0):.0f} pips
   • Trades: {result.get('trades', 0)}
   • Winrate: {result.get('winrate', 0):.1%}
   • Drawdown max: {result.get('drawdown', 0):.0f} pips
   • Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}
   • Profit Factor: {result.get('profit_factor', 0):.2f}
   • Score: {score:.1f}/100

🛡️ **Risk Management:**
   • SL: {rm.get('sl_pips', '?')} pips
   • TP: {rm.get('tp_pips', '?')} pips
   • Risk/Reward: {rm.get('risk_reward_ratio', '?')}
   • Risque/trade: {rm.get('risk_per_trade_pct', '?')}%
"""

    ao = search_result.get("anti_overfit")
    if ao:
        emoji = "✅" if ao.get("is_valid") else "⚠️"
        summary += f"""
🧪 **Validation Anti-Overfitting:**
   {emoji} Verdict: {ao.get('verdict', '?')}
   • Dégradation: {ao.get('degradation_ratio', 0):.2f}
   • Consistance: {ao.get('consistency_score', 0):.2f}
"""

    summary += f"""
🔧 Mode: {'MT5 réel' if search_result.get('mt5_connected') else 'Simulation'}
⚠️ *Les résultats passés ne garantissent pas les performances futures.*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary.strip()