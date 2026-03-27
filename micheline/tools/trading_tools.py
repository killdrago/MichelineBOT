"""
micheline/tools/trading_tools.py

Outils de trading pour l'agent Micheline.
Fonctionne UNIQUEMENT avec MT5 réel. Aucune simulation.
Supporte toutes les familles de stratégies avancées.
"""

import logging
import random
import time
import copy
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("micheline.tools.trading")

# ═══════════════════════════════════════
# Vérification MT5
# ═══════════════════════════════════════
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.error("❌ MetaTrader5 non installé — pip install MetaTrader5")


def _is_mt5_connected() -> bool:
    if not MT5_AVAILABLE:
        return False
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except Exception:
        return False


def _ensure_mt5_ready() -> bool:
    if not MT5_AVAILABLE:
        return False
    if _is_mt5_connected():
        return True
    try:
        if mt5.initialize():
            logger.info("✅ MT5 initialisé avec succès")
            return True
        else:
            error = mt5.last_error()
            logger.error(f"❌ MT5 initialize() échoué: {error}")
            return False
    except Exception as e:
        logger.error(f"❌ MT5 exception: {e}")
        return False


# ═══════════════════════════════════════
# INSTANCE ENGINE (singleton partagé)
# ═══════════════════════════════════════
_engine_instance = None


def _get_engine():
    global _engine_instance
    if _engine_instance is None:
        try:
            try:
                from micheline.trading.engine import TradingEngine
            except ImportError:
                import sys, os
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
            logger.warning(f"⚠️ TradingEngine non disponible: {e}")
            _engine_instance = None
    return _engine_instance


# ═══════════════════════════════════════
# GÉNÉRATION DE STRATÉGIE ALÉATOIRE
# ═══════════════════════════════════════

def generate_random_strategy(symbol: str = "EURUSD", timeframe: Optional[str] = None) -> dict:
    """
    Génère une stratégie de trading aléatoire.
    Essaie d'utiliser les templates avancés si disponibles,
    sinon utilise le générateur basique.
    """
    # Essayer les templates avancés
    try:
        try:
            from micheline.trading.strategies.strategy_templates import (
                get_all_strategy_families, generate_strategy_from_template
            )
        except ImportError:
            from trading.strategies.strategy_templates import (
                get_all_strategy_families, generate_strategy_from_template
            )

        families = get_all_strategy_families()
        family = random.choice(families)
        return generate_strategy_from_template(family, symbol, timeframe)

    except ImportError:
        logger.debug("Templates avancés non disponibles, utilisation du générateur basique")
    except Exception as e:
        logger.warning(f"Erreur templates: {e}, fallback basique")

    # Fallback : générateur basique
    return _generate_basic_strategy(symbol, timeframe)


def _generate_basic_strategy(symbol: str = "EURUSD", timeframe: Optional[str] = None) -> dict:
    """Générateur basique de stratégie (fallback)."""
    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]
    tf = timeframe if timeframe else random.choice(timeframes)

    entry_types = ["crossover", "threshold", "breakout", "momentum"]
    exit_types = ["opposite_signal", "trailing_stop", "time_based", "indicator_reversal"]

    tf_multiplier = {
        "M1": 0.7, "M5": 0.8, "M15": 0.9, "M30": 1.0,
        "H1": 1.2, "H4": 1.5, "D1": 2.0
    }.get(tf, 1.0)

    sl_ranges = {
        "M1": (5, 20), "M5": (8, 30), "M15": (10, 40), "M30": (15, 50),
        "H1": (20, 80), "H4": (30, 120), "D1": (50, 200)
    }
    sl_min, sl_max = sl_ranges.get(tf, (20, 80))

    indicator_pool = [
        {"type": "RSI", "params": {
            "period": max(7, int(random.randint(10, 21) * tf_multiplier)),
            "overbought": random.randint(65, 85),
            "oversold": random.randint(15, 35)
        }},
        {"type": "EMA", "params": {
            "period": max(5, int(random.randint(8, 50) * tf_multiplier))
        }},
        {"type": "SMA", "params": {
            "period": max(10, int(random.randint(15, 100) * tf_multiplier))
        }},
        {"type": "MACD", "params": {
            "fast": max(5, int(random.randint(8, 15) * tf_multiplier)),
            "slow": max(15, int(random.randint(20, 30) * tf_multiplier)),
            "signal": max(5, int(random.randint(7, 12) * tf_multiplier))
        }},
        {"type": "STOCH", "params": {
            "k_period": max(5, int(random.randint(8, 14) * tf_multiplier)),
            "d_period": random.randint(3, 5),
            "slowing": random.randint(1, 3),
            "overbought": random.randint(70, 85),
            "oversold": random.randint(15, 30)
        }},
        {"type": "BB", "params": {
            "period": max(10, int(random.randint(15, 25) * tf_multiplier)),
            "deviation": round(random.uniform(1.5, 3.0), 1)
        }},
        {"type": "ATR", "params": {
            "period": max(7, int(random.randint(10, 20) * tf_multiplier))
        }},
        {"type": "ADX", "params": {
            "period": max(7, int(random.randint(10, 20) * tf_multiplier)),
            "threshold": random.randint(20, 35)
        }},
    ]

    num_indicators = random.randint(2, 4)
    selected = random.sample(indicator_pool, min(num_indicators, len(indicator_pool)))

    sl = random.randint(sl_min, sl_max)
    tp_ratio = round(random.uniform(1.0, 5.0), 1)

    strategy = {
        "id": f"basic_{int(time.time())}_{random.randint(1000, 9999)}",
        "symbol": symbol,
        "timeframe": tf,
        "family": "basic_random",
        "indicators": selected,
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

def improve_strategy(strategy: dict, score: float) -> dict:
    """Crée une mutation d'une stratégie existante."""
    # Essayer le mutateur avancé
    try:
        try:
            from micheline.trading.strategies.strategy_templates import mutate_strategy
        except ImportError:
            from trading.strategies.strategy_templates import mutate_strategy
        return mutate_strategy(strategy, strength=0.2)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback basique
    mutated = copy.deepcopy(strategy)
    mutated["id"] = f"mut_{int(time.time())}_{random.randint(1000, 9999)}"

    if mutated.get("indicators"):
        for ind in mutated["indicators"]:
            if "params" in ind:
                for key, val in ind["params"].items():
                    if isinstance(val, int) and val > 0:
                        delta = max(1, int(val * 0.2))
                        ind["params"][key] = max(1, val + random.randint(-delta, delta))
                    elif isinstance(val, float) and val > 0:
                        delta = val * 0.2
                        ind["params"][key] = round(max(0.01, val + random.uniform(-delta, delta)), 3)

    rm = mutated.get("risk_management", {})
    if rm:
        sl = rm.get("stop_loss", 50)
        sl_delta = max(1, int(sl * 0.15))
        new_sl = max(5, sl + random.randint(-sl_delta, sl_delta))
        rm["stop_loss"] = new_sl
        rr = rm.get("risk_reward_ratio", 2.0)
        new_rr = round(max(0.5, rr + random.uniform(-0.5, 0.5)), 1)
        rm["risk_reward_ratio"] = new_rr
        rm["take_profit"] = int(new_sl * new_rr)

    return mutated


# ═══════════════════════════════════════
# BACKTEST — MT5 RÉEL UNIQUEMENT
# ═══════════════════════════════════════

def _make_error_result(error_msg: str) -> dict:
    return {
        "profit": 0.0, "trades": 0, "winrate": 0.0, "drawdown": 0.0,
        "sharpe_ratio": 0.0, "profit_factor": 0.0,
        "trade_results": [], "equity_curve": [0.0],
        "date_start": "N/A", "date_end": "N/A",
        "data_source": "erreur", "error": error_msg
    }


def run_backtest(strategy: dict, start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """
    Lance un backtest. Priorité :
    1. MT5 Bridge EA (résultats les plus fiables)
    2. Backtest Python avec données MT5 (fallback)
    """
    symbol = strategy.get("symbol", "EURUSD")
    timeframe = strategy.get("timeframe", "H1")

    if not MT5_AVAILABLE:
        return _make_error_result("MetaTrader5 non installé")
    if not _ensure_mt5_ready():
        return _make_error_result("MT5 non connecté")

    # ── Essayer le Bridge EA d'abord ──
    try:
        try:
            from micheline.trading.mt5_bridge import MT5Bridge
        except ImportError:
            from trading.mt5_bridge import MT5Bridge

        bridge = MT5Bridge(timeout=60)
        if bridge.is_ea_running():
            logger.info(f"🌉 Bridge EA détecté — backtest via MT5")
            result = bridge.run_backtest(strategy)
            if result and not result.get("error") and result.get("trades", 0) > 0:
                result["timeframe"] = timeframe
                result["symbol"] = symbol
                return result
            elif result and result.get("error"):
                logger.warning(f"Bridge erreur: {result['error']}, fallback Python")
            else:
                logger.warning("Bridge: 0 trades, fallback Python")
    except Exception as e:
        logger.warning(f"Bridge non disponible: {e}, fallback Python")

    # ── Fallback : backtest Python ──
    try:
        try:
            from micheline.trading.mt5_backtest import run_real_backtest
        except ImportError:
            from trading.mt5_backtest import run_real_backtest
    except ImportError as e:
        return _make_error_result(f"Module mt5_backtest introuvable: {e}")

    try:
        logger.info(f"🐍 Backtest Python: {symbol} {timeframe}")
        result = run_real_backtest(strategy, start=start, end=end)

        if result is None:
            return _make_error_result(f"Pas de données MT5 pour {symbol} {timeframe}")

        data_source = result.get("data_source", "unknown")
        if data_source in ("simulation", "simulated", "fake", "random"):
            return _make_error_result(f"Source '{data_source}' rejetée")

        return result

    except Exception as e:
        logger.error(f"❌ Erreur backtest: {e}", exc_info=True)
        return _make_error_result(str(e))

# ═══════════════════════════════════════
# RECHERCHE DE STRATÉGIE (standard)
# ═══════════════════════════════════════

def trading_search(params: dict) -> dict:
    """
    Recherche de stratégie par algorithme génétique.
    Utilise les templates avancés si disponibles.
    """
    try:
        if not _ensure_mt5_ready():
            return {
                "success": False,
                "error": "MT5 non disponible. Ouvrez MetaTrader 5 et connectez-vous.",
                "formatted": "❌ **MetaTrader 5 non disponible**\n\n"
                             "1. Ouvrez MetaTrader 5\n2. Connectez-vous\n3. Réessayez"
            }

        symbols = params.get("symbols", ["EURUSD"])
        pop_size = min(params.get("population_size", 10), 50)
        max_gen = min(params.get("max_generations", 3), 10)
        timeframes_param = params.get("timeframes", [])
        forced_timeframe = timeframes_param[0] if timeframes_param else None
        symbol = symbols[0] if symbols else "EURUSD"

        logger.info(
            f"=== RECHERCHE MT5: {symbol} | TF={forced_timeframe or 'auto'} | "
            f"pop={pop_size} | gen={max_gen} ==="
        )

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                def evaluate_strategy(r):
                    return max(0.0, min(100.0,
                        r.get("profit", 0) / 50.0 + r.get("winrate", 0) * 30.0 +
                        r.get("profit_factor", 0) * 5.0 - r.get("drawdown", 0) / 10.0))

        # Génération initiale
        population = []
        attempts = 0
        max_attempts = pop_size * 5
        errors_count = 0

        while len(population) < pop_size and attempts < max_attempts:
            attempts += 1
            strat = generate_random_strategy(symbol, timeframe=forced_timeframe)
            result = run_backtest(strat)

            if result.get("error"):
                errors_count += 1
                if errors_count >= 3:
                    return {
                        "success": False,
                        "error": f"MT5 erreurs répétées: {result['error']}",
                        "formatted": f"❌ Erreur MT5: {result['error']}"
                    }
                continue
            else:
                errors_count = 0

            trades = result.get("trades", 0)
            if trades < 5:
                continue

            score = evaluate_strategy(result)
            population.append({
                "strategy": strat, "result": result,
                "score": score, "generation": 0
            })
            logger.info(
                f"  Gen 0 | #{len(population)}: score={score:.1f} | "
                f"profit={result['profit']:.0f} | trades={trades} | "
                f"family={strat.get('family', '?')}"
            )

        if not population:
            return {
                "success": False,
                "error": f"Aucune stratégie valide pour {symbol}",
                "formatted": f"❌ Aucune stratégie valide ({attempts} tentatives)"
            }

        logger.info(f"  Population initiale: {len(population)} ({attempts} tentatives)")

        # Évolution
        total_evaluated = attempts
        for gen in range(1, max_gen + 1):
            population.sort(key=lambda x: x["score"], reverse=True)
            survivors = population[:max(2, pop_size // 2)]
            new_pop = list(survivors)
            child_attempts = 0

            while len(new_pop) < pop_size and child_attempts < pop_size * 3:
                child_attempts += 1
                total_evaluated += 1
                parent = random.choice(survivors)
                child = improve_strategy(parent["strategy"], parent["score"])
                if forced_timeframe:
                    child["timeframe"] = forced_timeframe

                child_result = run_backtest(child)
                if child_result.get("error") or child_result.get("trades", 0) < 5:
                    continue

                child_score = evaluate_strategy(child_result)
                new_pop.append({
                    "strategy": child, "result": child_result,
                    "score": child_score, "generation": gen
                })

            population = new_pop
            best = max(population, key=lambda x: x["score"])
            logger.info(
                f"  Gen {gen} | Best: score={best['score']:.1f} | "
                f"profit={best['result']['profit']:.0f}"
            )

        # Résultat
        population.sort(key=lambda x: x["score"], reverse=True)
        best = population[0]
        top_3 = population[:3]

        # Anti-overfit
        anti_overfit_result = None
        try:
            try:
                from micheline.trading.anti_overfit import AntiOverfitValidator
            except ImportError:
                from trading.anti_overfit import AntiOverfitValidator
            validator = AntiOverfitValidator(
                backtest_runner=lambda config, s, e: run_backtest(config, start=s, end=e),
                n_folds=3
            )
            test_result = run_backtest(best["strategy"])
            if not test_result.get("error"):
                report = validator.quick_validate(
                    train_metrics=best["result"], test_metrics=test_result,
                    strategy_id=best["strategy"]["id"]
                )
                anti_overfit_result = {
                    "verdict": report.verdict.value, "is_valid": report.is_valid,
                    "degradation_ratio": report.degradation_ratio,
                    "consistency_score": report.consistency_score
                }
        except Exception as e:
            logger.warning(f"Anti-overfit skipped: {e}")

        output = {
            "success": True, "symbol": symbol,
            "best_strategy": best["strategy"],
            "best_result": {k: v for k, v in best["result"].items()
                           if k not in ("trade_results", "equity_curve")},
            "best_score": best["score"],
            "generations": max_gen, "total_evaluated": total_evaluated,
            "top_3": [
                {
                    "id": p["strategy"]["id"], "score": p["score"],
                    "profit": p["result"]["profit"], "winrate": p["result"]["winrate"],
                    "drawdown": p["result"]["drawdown"],
                    "sharpe": p["result"].get("sharpe_ratio", 0),
                    "trades": p["result"]["trades"],
                    "family": p["strategy"].get("family", "?"),
                }
                for p in top_3
            ],
            "anti_overfit": anti_overfit_result,
            "mt5_connected": True, "mode": "mt5_real"
        }
        output["formatted"] = format_strategy_summary(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_search: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


# ═══════════════════════════════════════
# RECHERCHE EXHAUSTIVE (TOUTES STRATÉGIES)
# ═══════════════════════════════════════

def trading_exhaustive_search(params: dict) -> dict:
    """
    Recherche EXHAUSTIVE de stratégies.
    Teste TOUTES les familles (Ichimoku, ICT/SMC, Volume Profile, etc.)
    sur TOUS les timeframes avec de multiples configurations.
    """
    try:
        if not _ensure_mt5_ready():
            return {
                "success": False,
                "error": "MT5 non disponible",
                "formatted": "❌ MetaTrader 5 non disponible. Ouvrez MT5 et connectez-vous."
            }

        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        timeframes = params.get("timeframes", None)
        families = params.get("families", None)
        variants = params.get("variants_per_family", 3)
        mutations = params.get("mutations_per_best", 5)
        max_strats = params.get("max_strategies", 200)

        logger.info(
            f"🔍 RECHERCHE EXHAUSTIVE: {symbol} | "
            f"familles={'toutes' if not families else families} | "
            f"TF={'tous' if not timeframes else timeframes}"
        )

        try:
            from micheline.trading.strategies.exhaustive_search import ExhaustiveSearch
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.strategies.exhaustive_search import ExhaustiveSearch
                from trading.metrics import evaluate_strategy
            except ImportError:
                return {
                    "success": False,
                    "error": "Module exhaustive_search non trouvé",
                    "formatted": "❌ Module de recherche exhaustive non installé."
                }

        searcher = ExhaustiveSearch(
            run_backtest_fn=run_backtest,
            evaluate_fn=evaluate_strategy
        )

        result = searcher.search(
            symbol=symbol,
            timeframes=timeframes,
            families=families,
            variants_per_family=variants,
            mutations_per_best=mutations,
            max_total_strategies=max_strats,
        )

        if result.get("success"):
            result["formatted"] = _format_exhaustive_result(result)
        else:
            result["formatted"] = f"❌ {result.get('error', 'Erreur inconnue')}"

        return result

    except Exception as e:
        logger.error(f"Erreur trading_exhaustive_search: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_exhaustive_result(result: dict) -> str:
    """Formate le résultat d'une recherche exhaustive."""
    lines = []
    lines.append("🔍 **RECHERCHE EXHAUSTIVE TERMINÉE**")
    lines.append("━" * 50)

    symbol = result.get("symbol", "?")
    tested = result.get("total_tested", 0)
    valid = result.get("total_valid", 0)
    errors = result.get("total_errors", 0)
    elapsed = result.get("elapsed_formatted", "?")

    lines.append(f"📊 Symbole: {symbol}")
    lines.append(f"🔬 Stratégies testées: {tested}")
    lines.append(f"✅ Valides: {valid}")
    lines.append(f"❌ Erreurs: {errors}")
    lines.append(f"⏱️ Durée: {elapsed}")

    # Meilleure stratégie
    best = result.get("best_strategy", {})
    best_result = result.get("best_result", {})
    best_score = result.get("best_score", 0)
    best_family = result.get("best_family", "?")
    best_tf = result.get("best_timeframe", "?")
    best_variant = result.get("best_variant", "?")

    lines.append(f"\n🏆 **MEILLEURE STRATÉGIE:**")
    lines.append(f"  Famille: {best_family}")
    lines.append(f"  Variante: {best_variant}")
    lines.append(f"  Timeframe: {best_tf}")
    lines.append(f"  Score: {best_score:.1f}/100")
    lines.append(f"  Profit: {best_result.get('profit', 0):.0f} pips")
    lines.append(f"  Trades: {best_result.get('trades', 0)}")
    lines.append(f"  Winrate: {best_result.get('winrate', 0):.1%}")
    lines.append(f"  Drawdown: {best_result.get('drawdown', 0):.0f} pips")
    lines.append(f"  Sharpe: {best_result.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Profit Factor: {best_result.get('profit_factor', 0):.2f}")

    # Train/Test
    tts = best_result.get("train_test_split")
    if tts:
        deg = tts.get("degradation_ratio", 0)
        cons = tts.get("consistency_score", 0)
        if deg >= 0.7 and cons >= 0.7:
            verdict = "✅ ROBUSTE"
        elif deg >= 0.4:
            verdict = "⚠️ ACCEPTABLE"
        else:
            verdict = "❌ FRAGILE"
        lines.append(f"  {verdict} (dég={deg:.2f} cons={cons:.2f})")
        lines.append(f"  Train: {tts.get('train_profit', 0):.0f} pips | "
                     f"Test: {tts.get('test_profit', 0):.0f} pips")

    # Indicateurs
    indicators = best.get("indicators", [])
    if indicators:
        ind_names = [ind.get("type", "?") for ind in indicators]
        lines.append(f"  Indicateurs: {', '.join(ind_names)}")

    # Risk Management
    rm = best.get("risk_management", {})
    if rm:
        lines.append(f"  SL: {rm.get('stop_loss', '?')} | TP: {rm.get('take_profit', '?')} | "
                     f"RR: {rm.get('risk_reward_ratio', '?')}")

    # Top 10
    top_10 = result.get("top_10", [])
    if top_10:
        lines.append(f"\n🏅 **TOP 10:**")
        for t in top_10:
            rank = t.get("rank", "?")
            icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            lines.append(
                f"  {icon} {t.get('family', '?')}/{t.get('timeframe', '?')} | "
                f"Score: {t.get('score', 0):.1f} | "
                f"Profit: {t.get('profit', 0):.0f} | "
                f"WR: {t.get('winrate', 0):.1%} | "
                f"Trades: {t.get('trades', 0)}"
            )

    # Stats par famille
    family_stats = result.get("family_stats", {})
    if family_stats:
        lines.append(f"\n📊 **Performance par famille:**")
        sorted_fams = sorted(family_stats.items(),
                             key=lambda x: x[1].get("best_score", 0), reverse=True)
        for fam, stats in sorted_fams[:10]:
            lines.append(
                f"  • {fam}: best={stats.get('best_score', 0):.1f} | "
                f"avg={stats.get('avg_score', 0):.1f} | "
                f"count={stats.get('count', 0)}"
            )

    # Stats par timeframe
    tf_stats = result.get("timeframe_stats", {})
    if tf_stats:
        lines.append(f"\n⏱️ **Performance par timeframe:**")
        sorted_tfs = sorted(tf_stats.items(),
                            key=lambda x: x[1].get("best_score", 0), reverse=True)
        for tf, stats in sorted_tfs:
            lines.append(
                f"  • {tf}: best={stats.get('best_score', 0):.1f} | "
                f"avg={stats.get('avg_score', 0):.1f} | "
                f"count={stats.get('count', 0)}"
            )

    lines.append(f"\n🔧 Mode: MT5 réel")
    lines.append("⚠️ Les résultats passés ne garantissent pas les performances futures.")
    lines.append("━" * 50)

    return "\n".join(lines)


# ═══════════════════════════════════════
# GÉNÉRATION SIMPLE
# ═══════════════════════════════════════

def trading_generate(params: dict) -> dict:
    """Génère UNE stratégie et la backteste sur MT5 réel."""
    try:
        if not _ensure_mt5_ready():
            return {
                "success": False, "error": "MT5 non disponible",
                "formatted": "❌ MetaTrader 5 non disponible."
            }

        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        timeframes_param = params.get("timeframes", [])
        forced_tf = timeframes_param[0] if timeframes_param else None

        best_result = None
        best_strategy = None
        for attempt in range(10):
            strategy = generate_random_strategy(symbol, timeframe=forced_tf)
            result = run_backtest(strategy)
            if result.get("error"):
                if attempt >= 2:
                    return {"success": False, "error": result["error"],
                            "formatted": f"❌ Erreur MT5: {result['error']}"}
                continue
            if result.get("trades", 0) >= 5:
                best_result = result
                best_strategy = strategy
                break

        if best_result is None or best_strategy is None:
            return {"success": False, "error": "Aucune stratégie valide",
                    "formatted": "❌ Aucune stratégie valide trouvée"}

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                def evaluate_strategy(r):
                    return max(0.0, min(100.0, r.get("profit", 0) / 50.0))

        score = evaluate_strategy(best_result)

        output = {
            "success": True, "symbol": symbol,
            "best_strategy": best_strategy,
            "best_result": {k: v for k, v in best_result.items()
                           if k not in ("trade_results", "equity_curve")},
            "best_score": score, "generations": 0, "total_evaluated": 1,
            "mt5_connected": True, "mode": "mt5_real"
        }
        output["formatted"] = format_strategy_summary(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_generate: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════
# TEST RAPIDE (QUICK TEST)
# ═══════════════════════════════════════

def trading_quick_test(params: dict) -> dict:
    """Teste rapidement N stratégies aléatoires avec MT5 réel."""
    try:
        if not _ensure_mt5_ready():
            return {"success": False, "error": "MT5 non disponible",
                    "formatted": "❌ MetaTrader 5 non disponible."}

        count = min(params.get("count", 5), 50)
        symbol = params.get("symbol", "EURUSD")
        if "symbols" in params and isinstance(params["symbols"], list) and params["symbols"]:
            symbol = params["symbols"][0]

        logger.info(f"=== Quick Test MT5: {count} stratégies sur {symbol} ===")

        engine = _get_engine()
        if engine:
            try:
                result = engine.quick_test(count=count)
                result["success"] = True
                result["formatted"] = _format_quick_test_result(result, symbol)
                return result
            except Exception as e:
                logger.warning(f"Engine quick_test échoué: {e}")

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                def evaluate_strategy(r):
                    return max(0.0, min(100.0,
                        r.get("profit", 0) / 50.0 + r.get("winrate", 0) * 30.0))

        results = []
        errors = 0
        for i in range(count):
            strat = generate_random_strategy(symbol)
            bt_result = run_backtest(strat)
            if bt_result.get("error"):
                errors += 1
                if errors >= 3:
                    break
                continue
            score = evaluate_strategy(bt_result)
            results.append({
                "id": strat["id"], "strategy": strat,
                "result": {k: v for k, v in bt_result.items()
                           if k not in ("trade_results", "equity_curve")},
                "score": score,
                "family": strat.get("family", "?"),
            })
            logger.info(
                f"  Test #{len(results)}: score={score:.1f} | "
                f"profit={bt_result['profit']:.0f} | {strat.get('family', '?')}"
            )

        if not results:
            return {"success": False, "error": "Aucun backtest réussi",
                    "formatted": "❌ Aucun backtest réussi. Vérifiez MT5."}

        results.sort(key=lambda x: x["score"], reverse=True)

        output = {
            "success": True, "count": len(results), "symbol": symbol,
            "results": results,
            "best": results[0] if results else None,
            "best_score": results[0]["score"] if results else 0,
            "avg_score": round(sum(r["score"] for r in results) / max(len(results), 1), 1),
            "mt5_connected": True, "mode": "mt5_real"
        }
        output["formatted"] = _format_quick_test_result(output, symbol)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_quick_test: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_quick_test_result(result: dict, symbol: str = "EURUSD") -> str:
    lines = []
    lines.append(f"⚡ **TEST RAPIDE MT5 — {symbol}**")
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
        family = r.get("family", "?")

        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        lines.append(
            f"{icon} {family} | Score: {score:.1f} | Profit: {profit:.0f} | "
            f"WR: {wr:.1%} | Trades: {trades} | DD: {dd:.0f}"
        )

    avg = result.get("avg_score", 0)
    lines.append("━" * 40)
    lines.append(f"📊 Score moyen: {avg:.1f}/100")
    lines.append(f"🔧 Mode: MT5 réel")
    return "\n".join(lines)


# ═══════════════════════════════════════
# AMÉLIORATION DE STRATÉGIE EXISTANTE
# ═══════════════════════════════════════

def trading_improve(params: dict) -> dict:
    """Améliore une stratégie existante par mutations successives."""
    try:
        if not _ensure_mt5_ready():
            return {"success": False, "error": "MT5 non disponible",
                    "formatted": "❌ MetaTrader 5 non disponible."}

        iterations = min(params.get("iterations", 20), 100)
        symbol = params.get("symbol", "EURUSD")

        logger.info(f"=== Amélioration MT5: {iterations} itérations sur {symbol} ===")

        engine = _get_engine()
        if engine and engine.best_strategies:
            try:
                best_existing = engine.best_strategies[0]
                result = engine.improve_strategy(
                    strategy=best_existing.get("strategy", best_existing),
                    iterations=iterations,
                    mutation_strength=params.get("mutation_strength", 0.2)
                )
                result["success"] = True
                result["formatted"] = _format_improve_result(result)
                return result
            except Exception as e:
                logger.warning(f"Engine improve échoué: {e}")

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                def evaluate_strategy(r):
                    return max(0.0, min(100.0,
                        r.get("profit", 0) / 50.0 + r.get("winrate", 0) * 30.0))

        # Trouver une base valide
        base_strat = None
        base_result = None
        for _ in range(10):
            candidate = generate_random_strategy(symbol)
            candidate_result = run_backtest(candidate)
            if not candidate_result.get("error") and candidate_result.get("trades", 0) >= 5:
                base_strat = candidate
                base_result = candidate_result
                break

        if base_strat is None:
            return {"success": False, "error": "Impossible de trouver une base valide",
                    "formatted": "❌ Aucune stratégie de base trouvée."}

        base_score = evaluate_strategy(base_result)
        best_strat = base_strat
        best_result = base_result
        best_score = base_score
        improvements = 0
        history = [{"iteration": 0, "score": base_score}]

        for i in range(iterations):
            mutated = improve_strategy(best_strat, best_score)
            mut_result = run_backtest(mutated)
            if mut_result.get("error") or mut_result.get("trades", 0) < 3:
                history.append({"iteration": i + 1, "score": best_score})
                continue
            mut_score = evaluate_strategy(mut_result)
            if mut_score > best_score:
                best_strat = mutated
                best_result = mut_result
                best_score = mut_score
                improvements += 1
                logger.info(f"  Amélioration #{improvements}: score {best_score:.1f}")
            history.append({"iteration": i + 1, "score": best_score})

        output = {
            "success": True, "symbol": symbol,
            "original_score": base_score, "final_score": best_score,
            "improved": best_score > base_score,
            "improvement_pct": round((best_score - base_score) / max(base_score, 0.01) * 100, 1),
            "improvements_found": improvements, "iterations": iterations,
            "best_strategy": best_strat,
            "best_result": {k: v for k, v in best_result.items()
                           if k not in ("trade_results", "equity_curve")},
            "best_score": best_score, "history": history[-10:],
            "mt5_connected": True, "mode": "mt5_real"
        }
        output["formatted"] = _format_improve_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_improve: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_improve_result(result: dict) -> str:
    lines = []
    lines.append("🔧 **AMÉLIORATION DE STRATÉGIE (MT5 réel)**")
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
    lines.append(f"🔄 {count} améliorations en {iters} itérations")

    br = result.get("best_result", {})
    if br:
        lines.append(f"\n📊 **Meilleure:**")
        lines.append(f"  • Profit: {br.get('profit', 0):.0f} pips | "
                     f"WR: {br.get('winrate', 0):.1%} | "
                     f"DD: {br.get('drawdown', 0):.0f} | "
                     f"Trades: {br.get('trades', 0)}")

    lines.append(f"\n🔧 Mode: MT5 réel")
    return "\n".join(lines)


# ═══════════════════════════════════════
# RAPPORT DE SESSION
# ═══════════════════════════════════════

def trading_report(params: dict) -> dict:
    """Génère un rapport de la session de trading actuelle."""
    try:
        engine = _get_engine()
        if engine:
            try:
                report = engine.get_session_report()
                report["success"] = True
                report["formatted"] = _format_report_result(report)
                return report
            except Exception as e:
                logger.warning(f"Engine report échoué: {e}")

        mt5_status = "✅ Connecté" if _is_mt5_connected() else "❌ Non connecté"
        output = {
            "success": True, "session_active": engine is not None,
            "strategies_tested": 0, "best_score": 0, "total_time": 0,
            "best_strategies": [], "mt5_status": mt5_status,
            "message": f"MT5: {mt5_status}\nAucune session active. Lancez une recherche."
        }
        output["formatted"] = _format_report_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_report: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_report_result(result: dict) -> str:
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
        lines.append(f"⏱️ Durée: {int(duration//60)}m {int(duration%60)}s")

    top = result.get("top_5", result.get("best_strategies", []))
    if top:
        lines.append(f"\n🏅 **Top stratégies:**")
        for i, s in enumerate(top[:5], 1):
            score = s.get("score", 0)
            profit = s.get("profit", s.get("result", {}).get("profit", 0))
            sid = s.get("id", "?")
            lines.append(f"  {i}. [{sid[:20]}] Score: {score:.1f} | Profit: {profit:.0f}")
    return "\n".join(lines)


# ═══════════════════════════════════════
# TOP STRATÉGIES
# ═══════════════════════════════════════

def trading_top_strategies(params: dict) -> dict:
    """Retourne les meilleures stratégies trouvées."""
    try:
        count = min(params.get("count", 5), 20)

        engine = _get_engine()
        if engine:
            try:
                top = engine.get_top_strategies(count=count)
                if top:
                    output = {
                        "success": True, "count": len(top),
                        "strategies": top,
                        "best_score": top[0].get("score", 0) if top else 0,
                    }
                    output["formatted"] = _format_top_result(output)
                    return output
            except Exception as e:
                logger.warning(f"Engine get_top échoué: {e}")

        if not _ensure_mt5_ready():
            return {"success": False, "error": "MT5 non disponible",
                    "formatted": "❌ MT5 non disponible."}

        try:
            from micheline.trading.metrics import evaluate_strategy
        except ImportError:
            try:
                from trading.metrics import evaluate_strategy
            except ImportError:
                def evaluate_strategy(r):
                    return max(0.0, min(100.0,
                        r.get("profit", 0) / 50.0 + r.get("winrate", 0) * 30.0))

        strategies = []
        for _ in range(count * 2):
            strat = generate_random_strategy("EURUSD")
            result = run_backtest(strat)
            if result.get("error"):
                continue
            score = evaluate_strategy(result)
            strategies.append({
                "id": strat["id"], "strategy": strat,
                "result": {k: v for k, v in result.items()
                           if k not in ("trade_results", "equity_curve")},
                "score": score, "family": strat.get("family", "?"),
            })

        if not strategies:
            return {"success": False, "error": "Aucun backtest réussi",
                    "formatted": "❌ Aucun backtest réussi."}

        strategies.sort(key=lambda x: x["score"], reverse=True)
        top = strategies[:count]

        output = {
            "success": True, "count": len(top), "strategies": top,
            "best_score": top[0]["score"] if top else 0,
            "mt5_connected": True, "mode": "mt5_real"
        }
        output["formatted"] = _format_top_result(output)
        return output

    except Exception as e:
        logger.error(f"Erreur trading_top_strategies: {e}", exc_info=True)
        return {"success": False, "error": str(e), "formatted": f"❌ Erreur: {e}"}


def _format_top_result(result: dict) -> str:
    lines = []
    lines.append("🏆 **TOP STRATÉGIES (MT5 réel)**")
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
        family = s.get("family", "?")

        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        lines.append(f"\n{icon} **{family}** — {s.get('id', '?')[:25]}")
        lines.append(f"   Score: {score:.1f}/100")
        lines.append(f"   Profit: {profit:.0f} pips | WR: {wr:.1%} | Trades: {trades}")
        lines.append(f"   DD: {dd:.0f} pips | Sharpe: {sharpe:.2f}")
    return "\n".join(lines)


# ═══════════════════════════════════════
# FORMATAGE RÉSUMÉ STRATÉGIE
# ═══════════════════════════════════════

def format_strategy_summary(search_result: dict) -> str:
    """Formate un résumé complet d'une stratégie trouvée."""
    if not search_result or not search_result.get("success"):
        return search_result.get("error", "❌ Erreur inconnue")

    symbol = search_result.get("symbol", "?")
    strat = search_result.get("best_strategy", {})
    result = search_result.get("best_result", {})
    score = search_result.get("best_score", 0)
    ao = search_result.get("anti_overfit")
    mode = search_result.get("mode", "unknown")

    lines = []
    lines.append(f"📊 **STRATÉGIE TROUVÉE — {symbol}**")
    lines.append("━" * 40)
    lines.append(f"🆔 ID: {strat.get('id', '?')}")

    family = strat.get("family", strat.get("strategy_type", "?"))
    variant = strat.get("variant", "")
    if family != "?":
        lines.append(f"📦 Famille: {family}")
    if variant:
        lines.append(f"🔀 Variante: {variant}")

    lines.append(f"⏱️ Timeframe: {strat.get('timeframe', '?')}")

    date_start = result.get("date_start", "?")
    date_end = result.get("date_end", "?")
    if date_start != "?" and date_end != "?" and date_start != "N/A":
        lines.append(f"📅 Période: {date_start} → {date_end}")

    lines.append(f"📡 Source: {result.get('data_source', mode)}")
    bars = result.get("bars_count", 0)
    if bars:
        lines.append(f"📏 Barres: {bars:,}")
    spread = result.get("spread_pips", 0)
    if spread:
        lines.append(f"💱 Spread: {spread:.1f} pips")

    # Indicateurs
    indicators = strat.get("indicators", [])
    if indicators:
        ind_strs = []
        for ind in indicators:
            ind_type = ind.get("type", "?")
            params = ind.get("params", {})
            if params:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                ind_strs.append(f"{ind_type}({params_str})")
            else:
                ind_strs.append(ind_type)
        lines.append(f"📈 Indicateurs: {', '.join(ind_strs)}")

    lines.append(f"🎯 Entrée: {strat.get('entry_type', '?')}")
    lines.append(f"🚪 Sortie: {strat.get('exit_type', '?')}")

    # Résultats CAPITAL
    starting = result.get("starting_capital", 10000)
    ending = result.get("ending_capital", 0)
    profit_money = result.get("profit_money", 0)

    lines.append(f"\n💰 **Résultats (backtest MT5 réel):**")
    if starting and ending:
        lines.append(f"  • Capital départ: {starting:,.0f}$")
        lines.append(f"  • Capital fin: {ending:,.0f}$")
        pct_return = ((ending - starting) / starting * 100) if starting > 0 else 0
        icon = "📈" if profit_money > 0 else "📉"
        lines.append(f"  • {icon} Profit: {profit_money:+,.2f}$ ({pct_return:+.1f}%)")

    lines.append(f"  • Profit (pips): {result.get('profit', 0):.0f}")
    lines.append(f"  • Trades: {result.get('trades', 0)}")
    lines.append(f"  • Winrate: {result.get('winrate', 0):.1%}")

    dd_pct = result.get("drawdown_pct", 0)
    dd_money = result.get("drawdown", 0)
    if dd_pct:
        lines.append(f"  • Drawdown max: {dd_money:,.0f}$ ({dd_pct:.1f}%)")
    else:
        lines.append(f"  • Drawdown max: {dd_money:.0f}")

    lines.append(f"  • Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  • Profit Factor: {result.get('profit_factor', 0):.2f}")
    lines.append(f"  • Score: {score:.1f}/100")

    wins = result.get("wins", 0)
    losses_count = result.get("losses", 0)
    if wins or losses_count:
        lines.append(f"  • Gagnants: {wins} | Perdants: {losses_count}")
    avg_win = result.get("avg_win", 0)
    avg_loss = result.get("avg_loss", 0)
    if avg_win or avg_loss:
        lines.append(f"  • Gain moyen: {avg_win:+.2f}$ | Perte moyenne: {avg_loss:+.2f}$")

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

    # Train/Test
    tts = result.get("train_test_split")
    if tts:
        lines.append(f"\n🧪 **Validation Train/Test:**")
        lines.append(f"  📅 Split: {tts.get('split_date', '?')}")
        lines.append(
            f"  📊 Train: {tts.get('train_profit', 0):+.2f}$ | "
            f"{tts.get('train_trades', 0)} trades | "
            f"WR: {tts.get('train_winrate', 0):.1%}"
        )
        lines.append(
            f"  📊 Test:  {tts.get('test_profit', 0):+.2f}$ | "
            f"{tts.get('test_trades', 0)} trades | "
            f"WR: {tts.get('test_winrate', 0):.1%}"
        )

        deg = tts.get("degradation_ratio", 0)
        cons = tts.get("consistency_score", 0)
        if deg >= 0.7 and cons >= 0.7:
            verdict = "✅ ROBUSTE"
            detail = "Performe bien sur données inédites"
        elif deg >= 0.4 and cons >= 0.5:
            verdict = "⚠️ ACCEPTABLE"
            detail = "Légère dégradation"
        elif deg > 0:
            verdict = "⚠️ FRAGILE"
            detail = "Dégradation significative — overfitting probable"
        else:
            verdict = "❌ OVERFITTING"
            detail = "Ne fonctionne pas sur données inédites"
        lines.append(f"  {verdict}")
        lines.append(f"  • Dégradation: {deg:.2f} | Consistance: {cons:.2f}")
        lines.append(f"  💡 {detail}")

    # Anti-overfit module
    if ao:
        lines.append(f"\n🧪 **Anti-Overfitting (module):**")
        icon = "✅" if ao.get("is_valid") else "⚠️"
        lines.append(f"  {icon} Verdict: {ao.get('verdict', '?')}")

    # Top 3
    top_3 = search_result.get("top_3", [])
    if len(top_3) > 1:
        lines.append(f"\n🏅 **Top 3:**")
        for i, t in enumerate(top_3, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            fam = t.get("family", "")
            fam_str = f" [{fam}]" if fam else ""
            lines.append(
                f"  {icon} Score: {t.get('score', 0):.1f} | "
                f"Profit: {t.get('profit', 0):.0f} pips | "
                f"WR: {t.get('winrate', 0):.1%}{fam_str}"
            )

    lines.append(f"\n🔧 Mode: MT5 réel (indicateurs natifs + spread réel)")
    lines.append("⚠️ Résultats passés ≠ performances futures.")
    lines.append("━" * 40)

    return "\n".join(lines)