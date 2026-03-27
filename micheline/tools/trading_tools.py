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

# ═══════════════════════════════════════
# Vérification MT5
# ═══════════════════════════════════════
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
        return mt5.terminal_info() is not None
    except Exception:
        return False


# ═══════════════════════════════════════
# INSTANCE ENGINE (singleton partagé)
# ═══════════════════════════════════════
_engine_instance = None


def _get_engine():
    """Retourne l'instance partagée du TradingEngine."""
    global _engine_instance
    if _engine_instance is None:
        try:
            try:
                from micheline.trading.engine import TradingEngine
            except ImportError:
                import sys
                import os
                # Ajouter le chemin parent si nécessaire
                parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                from micheline.trading.engine import TradingEngine

            _engine_instance = TradingEngine(
                run_backtest_fn=run_backtest,
                config={
                    "population_size": 10,
                    "max_generations": 5,
                    "target_score": 75.0,
                    "mutation_strength": 0.3
                }
            )
            logger.info("✅ TradingEngine initialisé")
        except Exception as e:
            logger.warning(f"⚠️ TradingEngine non disponible (fallback actif): {e}")
            _engine_instance = None
    return _engine_instance

# ═══════════════════════════════════════
# GÉNÉRATION DE STRATÉGIE ALÉATOIRE
# ═══════════════════════════════════════

def generate_random_strategy(symbol="EURUSD", timeframe=None):
    """
    Génère une stratégie de trading aléatoire.
    Si timeframe est spécifié, l'utilise au lieu d'en choisir un aléatoire.
    Les paramètres des indicateurs sont adaptés au timeframe.
    """
    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    tf = timeframe if timeframe else random.choice(timeframes)

    entry_types = ["crossover", "threshold", "breakout", "momentum"]
    exit_types = ["opposite_signal", "trailing_stop", "time_based", "indicator_reversal"]

    # Adapter les paramètres au timeframe
    # M1/M5 = périodes courtes, D1 = périodes longues
    tf_multiplier = {
        "M1": 0.5, "M5": 0.7, "M15": 0.85, "M30": 1.0,
        "H1": 1.2, "H4": 1.5, "D1": 2.0
    }.get(tf, 1.0)

    # SL/TP adaptés au timeframe
    sl_ranges = {
        "M1": (5, 20), "M5": (8, 30), "M15": (10, 40), "M30": (15, 50),
        "H1": (20, 80), "H4": (30, 120), "D1": (50, 200)
    }
    sl_min, sl_max = sl_ranges.get(tf, (20, 80))

    indicators = []
    num_indicators = random.randint(2, 4)

    indicator_pool = [
        {"type": "RSI", "params": {
            "period": max(3, int(random.randint(5, 21) * tf_multiplier)),
            "overbought": random.randint(65, 85),
            "oversold": random.randint(15, 35)
        }},
        {"type": "EMA", "params": {
            "period": max(3, int(random.randint(5, 50) * tf_multiplier))
        }},
        {"type": "SMA", "params": {
            "period": max(5, int(random.randint(10, 100) * tf_multiplier))
        }},
        {"type": "MACD", "params": {
            "fast": max(3, int(random.randint(8, 15) * tf_multiplier)),
            "slow": max(10, int(random.randint(20, 30) * tf_multiplier)),
            "signal": max(3, int(random.randint(7, 12) * tf_multiplier))
        }},
        {"type": "STOCH", "params": {
            "k_period": max(3, int(random.randint(5, 14) * tf_multiplier)),
            "d_period": random.randint(3, 5),
            "slowing": random.randint(1, 3),
            "overbought": random.randint(70, 85),
            "oversold": random.randint(15, 30)
        }},
        {"type": "BB", "params": {
            "period": max(5, int(random.randint(10, 25) * tf_multiplier)),
            "deviation": round(random.uniform(1.5, 3.0), 1)
        }},
        {"type": "ATR", "params": {
            "period": max(5, int(random.randint(10, 20) * tf_multiplier))
        }},
        {"type": "ADX", "params": {
            "period": max(5, int(random.randint(10, 20) * tf_multiplier)),
            "threshold": random.randint(20, 35)
        }},
    ]

    selected = random.sample(indicator_pool, min(num_indicators, len(indicator_pool)))
    indicators = selected

    sl = random.randint(sl_min, sl_max)
    tp_ratio = round(random.uniform(1.5, 4.0), 1)

    strategy = {
        "id": f"strat_{int(time.time())}_{random.randint(1000, 9999)}_mut{random.randint(100, 999)}",
        "symbol": symbol,
        "timeframe": tf,
        "indicators": indicators,
        "entry_type": random.choice(entry_types),
        "exit_type": random.choice(exit_types),
        "risk_management": {
            "stop_loss": sl,
            "take_profit": int(sl * tp_ratio),
            "risk_reward_ratio": tp_ratio,
            "risk_per_trade": round(random.uniform(0.5, 2.0), 1)
        }
    }

    return strategy

# ═══════════════════════════════════════
# AMÉLIORATION DE STRATÉGIE (MUTATION)
# ═══════════════════════════════════════

def improve_strategy(strategy, score):
    """Crée une mutation d'une stratégie existante."""
    mutated = copy.deepcopy(strategy)
    mutated["id"] = f"strat_{int(time.time())}_{random.randint(1000, 9999)}_mut{random.randint(100, 999)}"

    # Muter les indicateurs
    if mutated.get("indicators"):
        for ind in mutated["indicators"]:
            if "params" in ind:
                for key, val in ind["params"].items():
                    if isinstance(val, int):
                        delta = max(1, int(val * 0.2))
                        ind["params"][key] = max(1, val + random.randint(-delta, delta))
                    elif isinstance(val, float):
                        delta = val * 0.2
                        ind["params"][key] = round(max(0.1, val + random.uniform(-delta, delta)), 2)

    # Muter le risk management
    rm = mutated.get("risk_management", {})
    if rm:
        sl = rm.get("stop_loss", 50)
        sl_delta = max(1, int(sl * 0.15))
        new_sl = max(10, sl + random.randint(-sl_delta, sl_delta))
        rm["stop_loss"] = new_sl

        ratio = rm.get("risk_reward_ratio", 2.0)
        new_ratio = round(max(1.0, ratio + random.uniform(-0.3, 0.3)), 1)
        rm["risk_reward_ratio"] = new_ratio
        rm["take_profit"] = int(new_sl * new_ratio)

    return mutated


# ═══════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════

def run_backtest(strategy, start=None, end=None):
    """
    Lance un backtest avec données MT5 RÉELLES.
    PAS de simulation fake. Si MT5 n'est pas disponible, retourne une erreur.
    """
    symbol = strategy.get("symbol", "EURUSD")
    timeframe = strategy.get("timeframe", "H1")

    try:
        try:
            from micheline.trading.mt5_backtest import run_real_backtest
        except ImportError:
            from trading.mt5_backtest import run_real_backtest

        logger.info(f"🔬 Backtest RÉEL: {symbol} {timeframe}")
        result = run_real_backtest(strategy, start=start, end=end)

        if result is not None:
            logger.info(f"✅ Backtest RÉEL: {result['profit']:.1f} pips | {result['trades']} trades")
            return result
        else:
            logger.error(f"❌ Backtest échoué: pas de données MT5 pour {symbol} {timeframe}")
            return {
                "profit": 0, "trades": 0, "winrate": 0, "drawdown": 0,
                "sharpe_ratio": 0, "profit_factor": 0,
                "trade_results": [], "equity_curve": [0],
                "date_start": "N/A", "date_end": "N/A",
                "data_source": "erreur",
                "error": f"Pas de données MT5 pour {symbol} {timeframe}"
            }

    except Exception as e:
        logger.error(f"❌ Erreur backtest: {e}")
        return {
            "profit": 0, "trades": 0, "winrate": 0, "drawdown": 0,
            "sharpe_ratio": 0, "profit_factor": 0,
            "trade_results": [], "equity_curve": [0],
            "date_start": "N/A", "date_end": "N/A",
            "data_source": "erreur",
            "error": str(e)
        }
        
# ═══════════════════════════════════════
# RECHERCHE DE STRATÉGIE
# ═══════════════════════════════════════

def trading_search(params):
    """
    Recherche la meilleure stratégie par algorithme génétique.
    Utilise UNIQUEMENT des données MT5 réelles.
    Rejette les stratégies avec 0 trades.
    """
    try:
        symbols = params.get("symbols", ["EURUSD"])
        pop_size = min(params.get("population_size", 10), 50)
        max_gen = min(params.get("max_generations", 3), 10)

        # RESPECTER le timeframe demandé
        timeframes_param = params.get("timeframes", [])
        forced_timeframe = timeframes_param[0] if timeframes_param else None

        symbol = symbols[0] if symbols else "EURUSD"

        # Pas de dates start/end — on laisse MT5 donner les dernières barres disponibles
        logger.info(f"=== Recherche: {symbol} | TF={forced_timeframe or 'auto'} | pop={pop_size} | gen={max_gen} ===")

        # Import evaluate_strategy
        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100,
                    r.get("profit", 0) / 50 + r.get("winrate", 0) * 30 +
                    r.get("profit_factor", 0) * 5 - r.get("drawdown", 0) / 10))

        # Génération initiale
        population = []
        attempts = 0
        max_attempts = pop_size * 5  # Maximum 5x les tentatives pour remplir la population

        while len(population) < pop_size and attempts < max_attempts:
            attempts += 1
            strat = generate_random_strategy(symbol, timeframe=forced_timeframe)
            result = run_backtest(strat)

            trades = result.get("trades", 0)
            profit = result.get("profit", 0)

            # REJETER les stratégies avec trop peu de trades
            if trades < 5:
                logger.debug(f"  Rejetée: {trades} trades (min 5)")
                continue

            score = evaluate_strategy(result)
            population.append({
                "strategy": strat,
                "result": result,
                "score": score,
                "generation": 0
            })
            logger.info(f"  Gen 0 | #{len(population)}: score={score:.1f} | "
                        f"profit={profit:.0f} | trades={trades} | wr={result.get('winrate', 0):.1%}")

        if not population:
            logger.error("Aucune stratégie valide trouvée en génération initiale")
            return {
                "success": False,
                "error": f"Aucune stratégie avec suffisamment de trades trouvée pour {symbol} {forced_timeframe or 'auto'}",
                "formatted": f"❌ Aucune stratégie valide trouvée pour {symbol} {forced_timeframe or ''}"
            }

        logger.info(f"  Population initiale: {len(population)} stratégies valides ({attempts} tentatives)")

        # Évolution
        for gen in range(1, max_gen + 1):
            population.sort(key=lambda x: x["score"], reverse=True)
            survivors = population[:max(2, pop_size // 2)]
            new_pop = list(survivors)

            child_attempts = 0
            max_child_attempts = pop_size * 3

            while len(new_pop) < pop_size and child_attempts < max_child_attempts:
                child_attempts += 1
                parent = random.choice(survivors)
                child = improve_strategy(parent["strategy"], parent["score"])
                # Forcer le même timeframe
                if forced_timeframe:
                    child["timeframe"] = forced_timeframe
                child_result = run_backtest(child)

                if child_result.get("trades", 0) < 5:
                    continue

                child_score = evaluate_strategy(child_result)
                new_pop.append({
                    "strategy": child,
                    "result": child_result,
                    "score": child_score,
                    "generation": gen
                })

            population = new_pop
            best = max(population, key=lambda x: x["score"])
            logger.info(f"  Gen {gen} | Best: score={best['score']:.1f} | "
                        f"profit={best['result']['profit']:.0f} | trades={best['result']['trades']}")

        # Résultat final
        population.sort(key=lambda x: x["score"], reverse=True)
        best = population[0]
        top_3 = population[:3]

        # Filtrer les résultats insuffisants
        min_trades = 10
        if best["result"].get("trades", 0) < min_trades:
            logger.warning(f"Meilleure stratégie: seulement {best['result']['trades']} trades")

        # Anti-overfit
        anti_overfit_result = None
        try:
            try:
                from micheline.trading.anti_overfit import AntiOverfitValidator
            except ImportError:
                from trading.anti_overfit import AntiOverfitValidator

            validator = AntiOverfitValidator(
                backtest_runner=lambda config, s, e: run_backtest(config),
                n_folds=3
            )
            # Test sur les dernières données
            test_result = run_backtest(best["strategy"])
            report = validator.quick_validate(
                train_metrics=best["result"],
                test_metrics=test_result,
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
            "best_result": {k: v for k, v in best["result"].items()
                           if k not in ("trade_results", "equity_curve")},
            "best_score": best["score"],
            "generations": max_gen,
            "total_evaluated": attempts + sum(1 for _ in range(max_gen)),
            "top_3": [
                {
                    "id": p["strategy"]["id"],
                    "score": p["score"],
                    "profit": p["result"]["profit"],
                    "winrate": p["result"]["winrate"],
                    "drawdown": p["result"]["drawdown"],
                    "sharpe": p["result"]["sharpe_ratio"],
                    "trades": p["result"]["trades"]
                }
                for p in top_3
            ],
            "anti_overfit": anti_overfit_result,
            "mt5_connected": MT5_AVAILABLE and _is_mt5_connected(),
            "mode": "mt5_real"
        }

        output["formatted"] = format_strategy_summary(output)
        logger.info(f"=== Recherche terminée: score={best['score']:.1f} | "
                    f"trades={best['result']['trades']} ===")
        return output

    except Exception as e:
        logger.error(f"Erreur trading_search: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}

# ═══════════════════════════════════════
# GÉNÉRATION SIMPLE (fallback)
# ═══════════════════════════════════════

def trading_generate(params):
    """Génère UNE stratégie simple et la backteste."""
    try:
        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        # Respecter le timeframe
        timeframes_param = params.get("timeframes", [])
        forced_tf = timeframes_param[0] if timeframes_param else None

        # Essayer jusqu'à trouver une stratégie avec des trades
        for attempt in range(10):
            strategy = generate_random_strategy(symbol, timeframe=forced_tf)
            result = run_backtest(strategy)

            if result.get("trades", 0) >= 5:
                break

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50))

        score = evaluate_strategy(result)

        output = {
            "success": True,
            "symbol": symbol,
            "best_strategy": strategy,
            "best_result": {k: v for k, v in result.items()
                           if k not in ("trade_results", "equity_curve")},
            "best_score": score,
            "generations": 0,
            "total_evaluated": 1,
            "mode": "mt5_real"
        }
        output["formatted"] = format_strategy_summary(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_generate: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ═══════════════════════════════════════
# TEST RAPIDE (QUICK TEST)
# ═══════════════════════════════════════

def trading_quick_test(params):
    """
    Teste rapidement N stratégies aléatoires et retourne les résultats.
    Utilise le TradingEngine si disponible, sinon fait le test manuellement.
    """
    try:
        count = min(params.get("count", 5), 50)
        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        logger.info(f"=== Quick Test: {count} stratégies sur {symbol} ===")

        # Essayer avec le TradingEngine
        engine = _get_engine()
        if engine:
            try:
                result = engine.quick_test(count=count)
                result["success"] = True
                result["formatted"] = _format_quick_test_result(result, symbol)
                return result
            except Exception as e:
                logger.warning(f"Engine quick_test échoué, fallback manuel: {e}")

        # Fallback manuel
        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50 + r.get("winrate", 0) * 30))

        results = []
        for i in range(count):
            strat = generate_random_strategy(symbol)
            bt_result = run_backtest(strat)
            score = evaluate_strategy(bt_result)
            results.append({
                "id": strat["id"],
                "strategy": strat,
                "result": {k: v for k, v in bt_result.items() if k not in ("trade_results", "equity_curve")},
                "score": score
            })
            logger.info(f"  Test #{i+1}: score={score:.1f} | profit={bt_result['profit']:.0f} | wr={bt_result['winrate']:.1%}")

        results.sort(key=lambda x: x["score"], reverse=True)

        output = {
            "success": True,
            "count": count,
            "symbol": symbol,
            "results": results,
            "best": results[0] if results else None,
            "best_score": results[0]["score"] if results else 0,
            "avg_score": round(sum(r["score"] for r in results) / max(len(results), 1), 1),
        }
        output["formatted"] = _format_quick_test_result(output, symbol)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_quick_test: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_quick_test_result(result, symbol="EURUSD"):
    """Formate le résultat d'un quick test."""
    lines = []
    lines.append(f"⚡ **TEST RAPIDE — {symbol}**")
    lines.append("━" * 40)

    results_list = result.get("results", [])
    if not results_list:
        lines.append("Aucun résultat.")
        return "\n".join(lines)

    for i, r in enumerate(results_list[:10], 1):
        score = r.get("score", 0)
        res = r.get("result", {})
        profit = res.get("profit", 0)
        wr = res.get("winrate", 0)
        trades = res.get("trades", 0)
        dd = res.get("drawdown", 0)

        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        lines.append(f"{icon} Score: {score:.1f} | Profit: {profit:.0f} | WR: {wr:.1%} | Trades: {trades} | DD: {dd:.0f}")

    avg = result.get("avg_score", 0)
    lines.append("━" * 40)
    lines.append(f"📊 Score moyen: {avg:.1f}/100")
    lines.append(f"📈 Meilleur: {results_list[0].get('score', 0):.1f}/100")
    lines.append(f"📉 Pire: {results_list[-1].get('score', 0):.1f}/100")

    return "\n".join(lines)


# ═══════════════════════════════════════
# AMÉLIORATION DE STRATÉGIE EXISTANTE
# ═══════════════════════════════════════

def trading_improve(params):
    """
    Améliore une stratégie existante par mutations successives.
    """
    try:
        iterations = min(params.get("iterations", 20), 100)
        mutation_strength = params.get("mutation_strength", 0.2)
        symbol = params.get("symbol", "EURUSD")

        logger.info(f"=== Amélioration: {iterations} itérations sur {symbol} ===")

        # Essayer avec le TradingEngine
        engine = _get_engine()
        if engine and engine.best_strategies:
            try:
                best_existing = engine.best_strategies[0]
                result = engine.improve_strategy(
                    strategy=best_existing.get("strategy", best_existing),
                    iterations=iterations,
                    mutation_strength=mutation_strength
                )
                result["success"] = True
                result["formatted"] = _format_improve_result(result)
                return result
            except Exception as e:
                logger.warning(f"Engine improve échoué, fallback: {e}")

        # Fallback : générer une base puis améliorer
        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50 + r.get("winrate", 0) * 30))

        base_strat = generate_random_strategy(symbol)
        base_result = run_backtest(base_strat)
        base_score = evaluate_strategy(base_result)

        best_strat = base_strat
        best_result = base_result
        best_score = base_score
        improvements = 0
        history = [{"iteration": 0, "score": base_score}]

        for i in range(iterations):
            mutated = improve_strategy(best_strat, best_score)
            mut_result = run_backtest(mutated)
            mut_score = evaluate_strategy(mut_result)

            if mut_score > best_score:
                best_strat = mutated
                best_result = mut_result
                best_score = mut_score
                improvements += 1
                logger.info(f"  Amélioration #{improvements}: score {best_score:.1f} (iter {i+1})")

            history.append({"iteration": i + 1, "score": best_score})

        output = {
            "success": True,
            "symbol": symbol,
            "original_score": base_score,
            "final_score": best_score,
            "improved": best_score > base_score,
            "improvement_pct": round((best_score - base_score) / max(base_score, 1) * 100, 1),
            "improvements_found": improvements,
            "iterations": iterations,
            "best_strategy": best_strat,
            "best_result": {k: v for k, v in best_result.items() if k not in ("trade_results", "equity_curve")},
            "best_score": best_score,
            "history": history[-10:],
        }
        output["formatted"] = _format_improve_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_improve: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_improve_result(result):
    """Formate le résultat d'une amélioration."""
    lines = []
    lines.append("🔧 **AMÉLIORATION DE STRATÉGIE**")
    lines.append("━" * 40)

    original = result.get("original_score", 0)
    final = result.get("final_score", 0)
    improved = result.get("improved", False)
    pct = result.get("improvement_pct", 0)
    count = result.get("improvements_found", 0)
    iters = result.get("iterations", 0)

    icon = "📈" if improved else "📊"
    lines.append(f"{icon} Score initial: {original:.1f}")
    lines.append(f"{icon} Score final: {final:.1f}")
    lines.append(f"{'✅' if improved else '➖'} Amélioré: {'Oui' if improved else 'Non'} ({pct:+.1f}%)")
    lines.append(f"🔄 {count} améliorations trouvées en {iters} itérations")

    br = result.get("best_result", {})
    if br:
        lines.append("")
        lines.append("📊 **Meilleure stratégie:**")
        lines.append(f"  • Profit: {br.get('profit', 0):.0f} pips")
        lines.append(f"  • Winrate: {br.get('winrate', 0):.1%}")
        lines.append(f"  • Drawdown: {br.get('drawdown', 0):.0f} pips")
        lines.append(f"  • Sharpe: {br.get('sharpe_ratio', 0):.2f}")

    return "\n".join(lines)


# ═══════════════════════════════════════
# RAPPORT DE SESSION
# ═══════════════════════════════════════

def trading_report(params):
    """
    Génère un rapport de la session de trading actuelle.
    """
    try:
        logger.info("=== Génération rapport trading ===")

        # Essayer avec le TradingEngine
        engine = _get_engine()
        if engine:
            try:
                report = engine.get_session_report()
                report["success"] = True
                report["formatted"] = _format_report_result(report)
                return report
            except Exception as e:
                logger.warning(f"Engine report échoué: {e}")

        # Fallback : rapport basique
        output = {
            "success": True,
            "session_active": engine is not None,
            "strategies_tested": 0,
            "best_score": 0,
            "total_time": 0,
            "best_strategies": [],
            "message": "Aucune session de trading active. Lancez d'abord une recherche de stratégie."
        }
        output["formatted"] = _format_report_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_report: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_report_result(result):
    """Formate un rapport de session."""
    lines = []
    lines.append("📋 **RAPPORT DE SESSION TRADING**")
    lines.append("━" * 40)

    if result.get("message"):
        lines.append(result["message"])
        return "\n".join(lines)

    tested = result.get("strategies_tested", result.get("total_tested", 0))
    best_score = result.get("best_score", 0)
    duration = result.get("total_time", result.get("duration", 0))

    lines.append(f"📊 Stratégies testées: {tested}")
    lines.append(f"🏆 Meilleur score: {best_score:.1f}/100")
    if duration:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lines.append(f"⏱️ Durée: {minutes}m {seconds}s")

    top = result.get("top_5", result.get("best_strategies", []))
    if top:
        lines.append("")
        lines.append("🏅 **Top stratégies:**")
        for i, s in enumerate(top[:5], 1):
            score = s.get("score", 0)
            profit = s.get("profit", s.get("result", {}).get("profit", 0))
            sid = s.get("id", "?")
            lines.append(f"  {i}. [{sid[:20]}] Score: {score:.1f} | Profit: {profit:.0f}")

    return "\n".join(lines)


# ═══════════════════════════════════════
# TOP STRATÉGIES
# ═══════════════════════════════════════

def trading_top_strategies(params):
    """
    Retourne les meilleures stratégies trouvées.
    """
    try:
        count = min(params.get("count", 5), 20)
        logger.info(f"=== Top {count} stratégies ===")

        # Essayer avec le TradingEngine
        engine = _get_engine()
        if engine:
            try:
                top = engine.get_top_strategies(count=count)
                if top:
                    output = {
                        "success": True,
                        "count": len(top),
                        "strategies": top,
                        "best_score": top[0].get("score", 0) if top else 0,
                    }
                    output["formatted"] = _format_top_result(output)
                    return output
            except Exception as e:
                logger.warning(f"Engine get_top échoué: {e}")

        # Fallback : générer et trier
        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                evaluate_strategy = lambda r: max(0, min(100, r.get("profit", 0) / 50 + r.get("winrate", 0) * 30))

        strategies = []
        for _ in range(count * 2):
            strat = generate_random_strategy("EURUSD")
            result = run_backtest(strat)
            score = evaluate_strategy(result)
            strategies.append({
                "id": strat["id"],
                "strategy": strat,
                "result": {k: v for k, v in result.items() if k not in ("trade_results", "equity_curve")},
                "score": score
            })

        strategies.sort(key=lambda x: x["score"], reverse=True)
        top = strategies[:count]

        output = {
            "success": True,
            "count": len(top),
            "strategies": top,
            "best_score": top[0]["score"] if top else 0,
        }
        output["formatted"] = _format_top_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_top_strategies: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_top_result(result):
    """Formate le classement des meilleures stratégies."""
    lines = []
    lines.append("🏆 **TOP STRATÉGIES**")
    lines.append("━" * 40)

    strategies = result.get("strategies", [])
    if not strategies:
        lines.append("Aucune stratégie trouvée.")
        return "\n".join(lines)

    for i, s in enumerate(strategies, 1):
        score = s.get("score", 0)
        res = s.get("result", {})
        profit = res.get("profit", 0)
        wr = res.get("winrate", 0)
        trades = res.get("trades", 0)
        dd = res.get("drawdown", 0)
        sharpe = res.get("sharpe_ratio", 0)
        sid = s.get("id", "?")

        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        lines.append(f"\n{icon} **{sid[:30]}**")
        lines.append(f"   Score: {score:.1f}/100")
        lines.append(f"   Profit: {profit:.0f} pips | WR: {wr:.1%} | Trades: {trades}")
        lines.append(f"   DD: {dd:.0f} pips | Sharpe: {sharpe:.2f}")

    return "\n".join(lines)


# ═══════════════════════════════════════
# FORMATAGE RÉSUMÉ STRATÉGIE
# ═══════════════════════════════════════

def format_strategy_summary(search_result):
    """Formate un résumé complet d'une stratégie trouvée."""
    if not search_result or not search_result.get("success"):
        return search_result.get("error", "❌ Erreur inconnue")

    symbol = search_result.get("symbol", "?")
    strat = search_result.get("best_strategy", {})
    result = search_result.get("best_result", {})
    score = search_result.get("best_score", 0)
    ao = search_result.get("anti_overfit")
    mode = search_result.get("mode", "simulation")

    lines = []
    lines.append(f"📊 **STRATÉGIE TROUVÉE — {symbol}**")
    lines.append("━" * 40)
    lines.append(f"🆔 ID: {strat.get('id', '?')}")
    lines.append(f"⏱️ Timeframe: {strat.get('timeframe', '?')}")

    # Période de backtest
    date_start = result.get("date_start", "?")
    date_end = result.get("date_end", "?")
    if date_start != "?" and date_end != "?":
        lines.append(f"📅 Période: {date_start} → {date_end}")

    # Indicateurs
    indicators = strat.get("indicators", [])
    if indicators:
        ind_strs = []
        for ind in indicators:
            params_str = ", ".join(f"{k}={v}" for k, v in ind.get("params", {}).items())
            ind_strs.append(f"{ind.get('type', '?')}({params_str})")
        lines.append(f"📈 Indicateurs: {', '.join(ind_strs)}")

    lines.append(f"🎯 Entrée: {strat.get('entry_type', '?')}")
    lines.append(f"🚪 Sortie: {strat.get('exit_type', '?')}")

    # Résultats
    lines.append(f"\n💰 **Résultats (backtest):**")
    lines.append(f"  • Profit: {result.get('profit', 0):.0f} pips")
    lines.append(f"  • Trades: {result.get('trades', 0)}")
    lines.append(f"  • Winrate: {result.get('winrate', 0):.1%}")
    lines.append(f"  • Drawdown max: {result.get('drawdown', 0):.0f} pips")
    lines.append(f"  • Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  • Profit Factor: {result.get('profit_factor', 0):.2f}")
    lines.append(f"  • Score: {score:.1f}/100")

    # Nombre de générations et stratégies évaluées
    gens = search_result.get("generations", 0)
    total_eval = search_result.get("total_evaluated", 0)
    if gens > 0 or total_eval > 0:
        lines.append(f"  • Générations: {gens} | Évaluées: {total_eval}")

    # Risk Management
    rm = strat.get("risk_management", {})
    if rm:
        lines.append(f"\n🛡️ **Risk Management:**")
        lines.append(f"  • SL: {rm.get('stop_loss', '?')} pips")
        lines.append(f"  • TP: {rm.get('take_profit', '?')} pips")
        lines.append(f"  • Risk/Reward: {rm.get('risk_reward_ratio', '?')}")
        lines.append(f"  • Risque/trade: {rm.get('risk_per_trade', '?')}%")

    # Anti-overfit
    if ao:
        lines.append(f"\n🧪 **Validation Anti-Overfitting:**")
        verdict = ao.get("verdict", "?")
        icon = "✅" if ao.get("is_valid") else "⚠️"
        lines.append(f"  {icon} Verdict: {verdict}")
        lines.append(f"  • Dégradation: {ao.get('degradation_ratio', 0):.2f}")
        lines.append(f"  • Consistance: {ao.get('consistency_score', 0):.2f}")

    # Top 3
    top_3 = search_result.get("top_3", [])
    if len(top_3) > 1:
        lines.append(f"\n🏅 **Top 3:**")
        for i, t in enumerate(top_3, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            lines.append(f"  {icon} Score: {t.get('score', 0):.1f} | Profit: {t.get('profit', 0):.0f} | WR: {t.get('winrate', 0):.1%}")

    lines.append(f"\n🔧 Mode: MT5 réel" if mode == "mt5" else f"\n🔧 Mode: Simulation")
    lines.append("⚠️ Les résultats passés ne garantissent pas les performances futures.")
    lines.append("━" * 40)

    return "\n".join(lines)