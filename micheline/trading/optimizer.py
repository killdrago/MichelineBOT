# micheline/trading/optimizer.py
"""
Optimiseur de stratégies de trading.
Boucle évolutive : générer → tester → évaluer → améliorer → recommencer
"""

import random
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from micheline.trading.strategy_generator import StrategyGenerator
from micheline.trading.metrics import evaluate_result, compute_score, is_strategy_viable

logger = logging.getLogger("micheline.trading.optimizer")


class StrategyOptimizer:

    def __init__(
        self,
        run_backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        population_size: int = 20,
        elite_count: int = 5,
        max_generations: int = 50,
        mutation_strength: float = 0.3,
        target_score: float = 75.0,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        store_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.run_backtest_fn = run_backtest_fn
        self.population_size = max(3, population_size)
        self.elite_count = min(elite_count, self.population_size - 1)
        self.elite_count = max(1, self.elite_count)
        self.max_generations = max(1, max_generations)
        self.mutation_strength = mutation_strength
        self.target_score = target_score
        self.store_fn = store_fn

        self.generator = StrategyGenerator(symbols=symbols, timeframes=timeframes)

        self.history: List[Dict[str, Any]] = []
        self.best_strategy: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        self.best_metrics: Dict[str, float] = {}
        self.current_generation: int = 0

    def run(self) -> Dict[str, Any]:
        logger.info(
            f"=== Démarrage optimisation : {self.population_size} stratégies "
            f"x {self.max_generations} générations ==="
        )

        population = self._generate_initial_population()
        total_tested = 0

        for gen in range(1, self.max_generations + 1):
            self.current_generation = gen
            logger.info(f"── Génération {gen}/{self.max_generations} ──")

            scored_population = []
            for strategy in population:
                try:
                    result = self._safe_backtest(strategy)
                    metrics = evaluate_result(result)
                    score = metrics["composite_score"]
                    total_tested += 1

                    scored_population.append({
                        "strategy": strategy,
                        "result": result,
                        "metrics": metrics,
                        "score": score,
                    })

                    logger.debug(
                        f"  Stratégie {strategy['id']} : "
                        f"score={score:.2f}, profit={metrics['profit']:.2f}, "
                        f"dd={metrics['drawdown']:.1f}%, trades={metrics['trades']}"
                    )

                    if self.store_fn:
                        try:
                            self.store_fn({
                                "type": "backtest_result",
                                "generation": gen,
                                "strategy_id": strategy["id"],
                                "score": score,
                                "metrics": metrics,
                                "strategy": strategy,
                                "timestamp": datetime.now().isoformat(),
                            })
                        except Exception:
                            pass

                except Exception as e:
                    logger.warning(f"  Erreur backtest stratégie {strategy.get('id', '?')}: {e}")
                    total_tested += 1
                    continue

            if not scored_population:
                logger.warning("  Aucune stratégie évaluée, régénération complète")
                population = self._generate_initial_population()
                continue

            scored_population.sort(key=lambda x: x["score"], reverse=True)

            gen_best = scored_population[0]
            gen_stats = self._generation_stats(scored_population)

            logger.info(
                f"  Meilleur score gen {gen}: {gen_best['score']:.2f} "
                f"(id: {gen_best['strategy']['id']})"
            )
            logger.info(
                f"  Stats gen: avg={gen_stats['avg_score']:.2f}, "
                f"viable={gen_stats['viable_count']}/{len(scored_population)}"
            )

            if gen_best["score"] > self.best_score:
                self.best_score = gen_best["score"]
                self.best_strategy = gen_best["strategy"]
                self.best_metrics = gen_best["metrics"]
                logger.info(f"  ★ Nouveau meilleur global : {self.best_score:.2f}")

            self.history.append({
                "generation": gen,
                "best_score": gen_best["score"],
                "avg_score": gen_stats["avg_score"],
                "best_id": gen_best["strategy"]["id"],
                "viable_count": gen_stats["viable_count"],
                "population_size": len(scored_population),
            })

            if self.best_score >= self.target_score:
                logger.info(f"  🎯 Score cible atteint ! Arrêt génération {gen}")
                break

            if gen < self.max_generations:
                population = self._evolve(scored_population)

        result = {
            "best_strategy": self.best_strategy,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "generations_run": self.current_generation,
            "total_strategies_tested": total_tested,
            "history": self.history,
        }

        logger.info(
            f"=== Optimisation terminée : score = {self.best_score:.2f} "
            f"après {self.current_generation} gens, {total_tested} testées ==="
        )

        return result

    def _safe_backtest(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self.run_backtest_fn(strategy)
            if not isinstance(result, dict):
                raise ValueError(f"Backtest a retourné {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Backtest échoué pour {strategy.get('id', '?')}: {e}")
            return {
                "profit": 0.0, "drawdown": 100.0,
                "trades": 0, "winrate": 0.0, "error": str(e),
            }

    def _generate_initial_population(self) -> List[Dict[str, Any]]:
        pop = []
        for _ in range(self.population_size):
            try:
                pop.append(self.generator.generate_strategy())
            except Exception as e:
                logger.warning(f"Erreur génération stratégie: {e}")
        if not pop:
            pop.append(self.generator.generate_strategy())
        return pop

    def _evolve(self, scored_population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        next_gen = []
        pop_size = max(3, self.population_size)

        # 1. Élitisme
        elite_count = min(self.elite_count, len(scored_population))
        elites = scored_population[:elite_count]
        for elite in elites:
            next_gen.append(elite["strategy"])

        # 2. Mutations
        mutations_count = max(1, pop_size // 3)
        for _ in range(mutations_count):
            if elites:
                parent = random.choice(elites)["strategy"]
                try:
                    child = self.generator.mutate_strategy(parent, self.mutation_strength)
                    next_gen.append(child)
                except Exception:
                    pass

        # 3. Croisements
        crossover_count = max(1, pop_size // 4)
        viable = [sp["strategy"] for sp in scored_population if is_strategy_viable(sp["metrics"])]
        if len(viable) < 2:
            viable = [sp["strategy"] for sp in scored_population[:min(3, len(scored_population))]]

        for _ in range(crossover_count):
            if len(viable) >= 2:
                try:
                    parents = random.sample(viable, 2)
                    child = self.generator.crossover(parents[0], parents[1])
                    next_gen.append(child)
                except Exception:
                    pass

        # 4. Nouvelles stratégies
        while len(next_gen) < pop_size:
            try:
                next_gen.append(self.generator.generate_strategy())
            except Exception:
                break

        return next_gen[:pop_size]

    def _generation_stats(self, scored_population: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [sp["score"] for sp in scored_population]
        viable_count = sum(1 for sp in scored_population if is_strategy_viable(sp["metrics"]))
        return {
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "viable_count": viable_count,
        }