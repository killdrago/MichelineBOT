"""
micheline/trading/strategies/exhaustive_search.py

Moteur de recherche exhaustif.
Teste TOUTES les familles de stratégies, sur TOUS les timeframes,
avec de multiples configurations, et retourne les meilleures.
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger("micheline.trading.exhaustive")


class ExhaustiveSearch:
    """
    Recherche exhaustive de stratégies.
    Explore toutes les familles × timeframes × paramètres.
    """

    def __init__(self, run_backtest_fn: Callable, evaluate_fn: Callable):
        self.run_backtest = run_backtest_fn
        self.evaluate = evaluate_fn
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        self.start_time = 0
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def search(self, symbol: str = "EURUSD",
               timeframes: Optional[List[str]] = None,
               families: Optional[List[str]] = None,
               variants_per_family: int = 3,
               mutations_per_best: int = 5,
               min_trades: int = 10,
               max_total_strategies: int = 200,
               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Lance une recherche exhaustive.

        Args:
            symbol: paire à tester
            timeframes: liste des TF (None = tous)
            families: liste des familles (None = toutes)
            variants_per_family: nombre de variantes par famille×TF
            mutations_per_best: mutations sur les meilleures
            min_trades: minimum de trades pour accepter
            max_total_strategies: limite totale
            progress_callback: fn(current, total, best_score, message)

        Returns:
            dict avec résultats complets
        """
        from micheline.trading.strategies.strategy_templates import (
            get_all_strategy_families,
            generate_strategy_from_template,
            mutate_strategy
        )

        self._cancelled = False
        self.start_time = time.time()
        self.results = []
        self.errors = []

        # Configuration
        all_families = families or get_all_strategy_families()
        all_timeframes = timeframes or ["M5", "M15", "M30", "H1", "H4", "D1"]

        total_planned = len(all_families) * len(all_timeframes) * variants_per_family
        total_planned = min(total_planned, max_total_strategies)

        logger.info(
            f"{'='*60}\n"
            f"🔍 RECHERCHE EXHAUSTIVE\n"
            f"   Symbole: {symbol}\n"
            f"   Familles: {len(all_families)}\n"
            f"   Timeframes: {all_timeframes}\n"
            f"   Variantes/famille/TF: {variants_per_family}\n"
            f"   Max stratégies: {max_total_strategies}\n"
            f"{'='*60}"
        )

        tested = 0
        best_score = 0
        consecutive_errors = 0

        # ═══ PHASE 1 : Explorer toutes les familles ═══
        logger.info("━━━ PHASE 1: Exploration de toutes les familles ━━━")

        for family in all_families:
            if self._cancelled:
                break

            for tf in all_timeframes:
                if self._cancelled or tested >= max_total_strategies:
                    break

                for v in range(variants_per_family):
                    if self._cancelled or tested >= max_total_strategies:
                        break

                    tested += 1

                    try:
                        strat = generate_strategy_from_template(family, symbol, tf)
                        result = self.run_backtest(strat)

                        if result.get("error"):
                            consecutive_errors += 1
                            self.errors.append({
                                "family": family, "tf": tf,
                                "error": result["error"]
                            })
                            if consecutive_errors >= 5:
                                logger.warning(f"⚠️ {consecutive_errors} erreurs consécutives, skip TF {tf}")
                                break
                            continue
                        else:
                            consecutive_errors = 0

                        trades = result.get("trades", 0)
                        if trades < min_trades:
                            continue

                        score = self.evaluate(result)

                        entry = {
                            "strategy": strat,
                            "result": {k: v for k, v in result.items()
                                       if k not in ("trade_results", "equity_curve")},
                            "full_result": result,
                            "score": score,
                            "family": family,
                            "timeframe": tf,
                            "variant": strat.get("variant", ""),
                            "phase": "exploration",
                        }
                        self.results.append(entry)

                        if score > best_score:
                            best_score = score
                            logger.info(
                                f"  🏆 NOUVEAU BEST: {family}/{tf}/{strat.get('variant', '')} "
                                f"score={score:.1f} | profit={result['profit']:.0f} | "
                                f"trades={trades} | WR={result['winrate']:.1%}"
                            )
                        else:
                            logger.debug(
                                f"  #{tested}: {family}/{tf} score={score:.1f} "
                                f"profit={result['profit']:.0f}"
                            )

                        if progress_callback:
                            progress_callback(
                                tested, total_planned, best_score,
                                f"Phase 1: {family}/{tf} v{v+1} — score={score:.1f}"
                            )

                    except Exception as e:
                        logger.warning(f"  Erreur {family}/{tf}: {e}")
                        self.errors.append({"family": family, "tf": tf, "error": str(e)})

        if not self.results:
            elapsed = time.time() - self.start_time
            return {
                "success": False,
                "error": "Aucune stratégie valide trouvée",
                "tested": tested,
                "elapsed": round(elapsed, 1),
                "errors": len(self.errors),
            }

        # ═══ PHASE 2 : Optimiser les meilleures ═══
        logger.info(f"\n━━━ PHASE 2: Optimisation des top {min(10, len(self.results))} stratégies ━━━")

        self.results.sort(key=lambda x: x["score"], reverse=True)
        top_to_optimize = self.results[:min(10, len(self.results))]

        for i, parent_entry in enumerate(top_to_optimize):
            if self._cancelled or tested >= max_total_strategies:
                break

            parent = parent_entry["strategy"]
            parent_score = parent_entry["score"]

            logger.info(
                f"  Optimisation #{i+1}: {parent.get('family', '?')}/{parent.get('timeframe', '?')} "
                f"(score={parent_score:.1f})"
            )

            for m in range(mutations_per_best):
                if self._cancelled or tested >= max_total_strategies:
                    break

                tested += 1

                try:
                    mutated = mutate_strategy(parent, strength=random.uniform(0.1, 0.3))
                    result = self.run_backtest(mutated)

                    if result.get("error") or result.get("trades", 0) < min_trades:
                        continue

                    score = self.evaluate(result)

                    entry = {
                        "strategy": mutated,
                        "result": {k: v for k, v in result.items()
                                   if k not in ("trade_results", "equity_curve")},
                        "full_result": result,
                        "score": score,
                        "family": mutated.get("family", parent.get("family", "mutated")),
                        "timeframe": mutated.get("timeframe", ""),
                        "variant": f"optimized_from_{parent.get('variant', '')}",
                        "phase": "optimization",
                    }
                    self.results.append(entry)

                    if score > best_score:
                        best_score = score
                        logger.info(
                            f"    🏆 AMÉLIORATION: score={score:.1f} (was {parent_score:.1f})"
                        )

                    if progress_callback:
                        progress_callback(
                            tested, max_total_strategies, best_score,
                            f"Phase 2: Optimisation #{i+1} mut {m+1}/{mutations_per_best}"
                        )

                except Exception as e:
                    logger.warning(f"    Mutation erreur: {e}")

        # ═══ RÉSULTAT FINAL ═══
        elapsed = time.time() - self.start_time
        self.results.sort(key=lambda x: x["score"], reverse=True)

        best = self.results[0]
        top_10 = self.results[:10]

        # Statistiques par famille
        family_stats = {}
        for r in self.results:
            fam = r.get("family", "unknown")
            if fam not in family_stats:
                family_stats[fam] = {"count": 0, "best_score": 0, "avg_score": 0, "scores": []}
            family_stats[fam]["count"] += 1
            family_stats[fam]["scores"].append(r["score"])
            family_stats[fam]["best_score"] = max(family_stats[fam]["best_score"], r["score"])

        for fam in family_stats:
            scores = family_stats[fam]["scores"]
            family_stats[fam]["avg_score"] = round(sum(scores) / len(scores), 1)
            del family_stats[fam]["scores"]

        # Statistiques par timeframe
        tf_stats = {}
        for r in self.results:
            tf = r.get("timeframe", "unknown")
            if tf not in tf_stats:
                tf_stats[tf] = {"count": 0, "best_score": 0, "scores": []}
            tf_stats[tf]["count"] += 1
            tf_stats[tf]["scores"].append(r["score"])
            tf_stats[tf]["best_score"] = max(tf_stats[tf]["best_score"], r["score"])

        for tf in tf_stats:
            scores = tf_stats[tf]["scores"]
            tf_stats[tf]["avg_score"] = round(sum(scores) / len(scores), 1)
            del tf_stats[tf]["scores"]

        output = {
            "success": True,
            "symbol": symbol,
            "total_tested": tested,
            "total_valid": len(self.results),
            "total_errors": len(self.errors),
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_formatted": f"{int(elapsed//60)}m {int(elapsed%60)}s",
            "best_strategy": best["strategy"],
            "best_result": best["result"],
            "best_score": best["score"],
            "best_family": best.get("family", "?"),
            "best_timeframe": best.get("timeframe", "?"),
            "best_variant": best.get("variant", "?"),
            "top_10": [
                {
                    "rank": i + 1,
                    "id": r["strategy"]["id"],
                    "family": r.get("family", "?"),
                    "timeframe": r.get("timeframe", "?"),
                    "variant": r.get("variant", "?"),
                    "score": r["score"],
                    "profit": r["result"].get("profit", 0),
                    "trades": r["result"].get("trades", 0),
                    "winrate": r["result"].get("winrate", 0),
                    "drawdown": r["result"].get("drawdown", 0),
                    "sharpe": r["result"].get("sharpe_ratio", 0),
                    "profit_factor": r["result"].get("profit_factor", 0),
                    "phase": r.get("phase", "?"),
                }
                for i, r in enumerate(top_10)
            ],
            "family_stats": family_stats,
            "timeframe_stats": tf_stats,
            "mode": "mt5_real",
        }

        logger.info(
            f"\n{'='*60}\n"
            f"🏆 RECHERCHE EXHAUSTIVE TERMINÉE\n"
            f"   Testées: {tested} | Valides: {len(self.results)} | Erreurs: {len(self.errors)}\n"
            f"   Durée: {output['elapsed_formatted']}\n"
            f"   Best: {best['family']}/{best['timeframe']} score={best['score']:.1f}\n"
            f"   Profit: {best['result']['profit']:.0f} pips | "
            f"Trades: {best['result']['trades']} | WR: {best['result']['winrate']:.1%}\n"
            f"{'='*60}"
        )

        return output