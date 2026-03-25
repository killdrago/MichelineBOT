# micheline/trading/engine.py
"""
Trading Engine — Point d'entrée principal du module trading.
Orchestre le générateur, l'optimiseur et les backtests.
Fait le lien entre le module trading et le reste de Micheline.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from micheline.trading.strategy_generator import StrategyGenerator
from micheline.trading.optimizer import StrategyOptimizer
from micheline.trading.metrics import evaluate_result, compute_score, is_strategy_viable, compare_strategies

logger = logging.getLogger("micheline.trading.engine")


class TradingEngine:
    """
    Moteur de trading principal.

    Responsabilités :
        - Interface simplifiée pour l'agent
        - Gestion de la recherche de stratégies
        - Suivi des meilleures stratégies
        - Rapports de résultats
    """

    def __init__(
        self,
        run_backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        store_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        retrieve_fn: Optional[Callable[[str], Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.run_backtest_fn = run_backtest_fn
        self.store_fn = store_fn
        self.retrieve_fn = retrieve_fn

        self.config = config or {}
        self.default_population = self.config.get("population_size", 20)
        self.default_generations = self.config.get("max_generations", 50)
        self.default_target_score = self.config.get("target_score", 75.0)
        self.default_mutation = self.config.get("mutation_strength", 0.3)

        self.best_strategies: List[Dict[str, Any]] = []
        self.total_runs: int = 0
        self.session_start = datetime.now()

        self.generator = StrategyGenerator(
            symbols=self.config.get("symbols"),
            timeframes=self.config.get("timeframes"),
        )

    def search_strategy(
        self,
        population_size: Optional[int] = None,
        max_generations: Optional[int] = None,
        target_score: Optional[float] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Lance une recherche complète de stratégie optimale."""
        # Valeurs par défaut robustes
        pop_size = max(3, int(population_size or self.default_population))
        max_gens = max(1, int(max_generations or self.default_generations))
        tgt_score = float(target_score or self.default_target_score)

        # Limiter pour éviter les recherches trop longues
        pop_size = min(pop_size, 50)
        max_gens = min(max_gens, 100)

        self.total_runs += 1
        run_id = f"run_{self.total_runs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"=== Recherche #{run_id} : {pop_size} pop x {max_gens} gens ===")

        elite_count = max(2, pop_size // 4)
        # S'assurer que elite_count < population_size
        elite_count = min(elite_count, pop_size - 1)

        optimizer = StrategyOptimizer(
            run_backtest_fn=self.run_backtest_fn,
            population_size=pop_size,
            elite_count=elite_count,
            max_generations=max_gens,
            mutation_strength=self.default_mutation,
            target_score=tgt_score,
            symbols=symbols,
            timeframes=timeframes,
            store_fn=self.store_fn,
        )

        try:
            result = optimizer.run()
        except Exception as e:
            logger.error(f"Erreur optimisation: {e}")
            import traceback
            traceback.print_exc()
            return {
                "run_id": run_id,
                "success": False,
                "best_strategy": None,
                "best_metrics": {},
                "best_score": 0.0,
                "generations_run": 0,
                "total_strategies_tested": 0,
                "convergence_history": [],
                "error": str(e),
            }

        if result.get("best_strategy"):
            self._register_best_strategy(result)

        if self.store_fn:
            try:
                self.store_fn({
                    "type": "optimization_run",
                    "run_id": run_id,
                    "best_score": result["best_score"],
                    "best_strategy_id": result["best_strategy"]["id"]
                    if result["best_strategy"] else None,
                    "generations_run": result["generations_run"],
                    "total_tested": result["total_strategies_tested"],
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"Store échoué: {e}")

        return {
            "run_id": run_id,
            "success": result["best_score"] > 0,
            "best_strategy": result["best_strategy"],
            "best_metrics": result.get("best_metrics", {}),
            "best_score": result["best_score"],
            "generations_run": result["generations_run"],
            "total_strategies_tested": result["total_strategies_tested"],
            "convergence_history": result["history"],
        }
        
    def quick_test(self, count: int = 5) -> Dict[str, Any]:
        """Test rapide de quelques stratégies sans optimisation."""
        logger.info(f"Quick test : {count} stratégies")
        results = []

        for i in range(count):
            strategy = self.generator.generate_strategy()
            try:
                backtest_result = self.run_backtest_fn(strategy)
                metrics = evaluate_result(backtest_result)
                results.append({
                    "strategy_id": strategy["id"],
                    "symbol": strategy["symbol"],
                    "timeframe": strategy["timeframe"],
                    "score": metrics["composite_score"],
                    "profit": metrics["profit"],
                    "drawdown": metrics["drawdown"],
                    "trades": metrics["trades"],
                    "viable": is_strategy_viable(metrics),
                    "strategy": strategy,
                    "date_start": backtest_result.get("date_start", ""),
                    "date_end": backtest_result.get("date_end", ""),
                })
            except Exception as e:
                results.append({
                    "strategy_id": strategy["id"],
                    "error": str(e),
                    "score": 0.0,
                })

        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "tested": count,
            "viable": sum(1 for r in results if r.get("viable", False)),
            "best_score": results[0]["score"] if results else 0,
            "results": results,
        }
        
    def evaluate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue une stratégie unique."""
        try:
            result = self.run_backtest_fn(strategy)
            metrics = evaluate_result(result)
            return {
                "strategy_id": strategy.get("id", "unknown"),
                "backtest_result": result,
                "metrics": metrics,
                "score": metrics["composite_score"],
                "viable": is_strategy_viable(metrics),
            }
        except Exception as e:
            logger.error(f"Erreur évaluation stratégie: {e}")
            return {
                "strategy_id": strategy.get("id", "unknown"),
                "error": str(e),
                "score": 0.0,
                "viable": False,
            }

    def improve_strategy(
        self,
        strategy: Dict[str, Any],
        iterations: int = 20,
        mutation_strength: float = 0.2,
    ) -> Dict[str, Any]:
        """Améliore une stratégie par mutations successives."""
        logger.info(f"Amélioration de {strategy.get('id', '?')} ({iterations} itérations)")

        base_eval = self.evaluate_strategy(strategy)
        best = {
            "strategy": strategy,
            "score": base_eval["score"],
            "metrics": base_eval.get("metrics", {}),
        }

        improvements = 0

        for i in range(iterations):
            mutated = self.generator.mutate_strategy(strategy, mutation_strength)
            try:
                result = self.run_backtest_fn(mutated)
                metrics = evaluate_result(result)
                score = metrics["composite_score"]

                if score > best["score"]:
                    best = {"strategy": mutated, "score": score, "metrics": metrics}
                    improvements += 1
                    logger.info(f"  Amélioration #{improvements} : score {best['score']:.2f} (iter {i+1})")
            except Exception as e:
                logger.debug(f"  Mutation {i+1} échouée: {e}")

        return {
            "original_score": base_eval["score"],
            "best_score": best["score"],
            "improved": best["score"] > base_eval["score"],
            "improvement_pct": round(
                ((best["score"] - base_eval["score"]) / max(base_eval["score"], 0.01)) * 100, 2
            ),
            "improvements_found": improvements,
            "iterations": iterations,
            "best_strategy": best["strategy"],
            "best_metrics": best["metrics"],
        }

    def get_top_strategies(self, count: int = 5) -> List[Dict[str, Any]]:
        """Retourne les N meilleures stratégies de la session."""
        sorted_strategies = sorted(
            self.best_strategies, key=lambda x: x["score"], reverse=True
        )
        return sorted_strategies[:count]

    def get_session_report(self) -> Dict[str, Any]:
        """Génère un rapport résumé de la session de trading."""
        duration = (datetime.now() - self.session_start).total_seconds()
        return {
            "session_start": self.session_start.isoformat(),
            "duration_seconds": round(duration, 1),
            "total_optimization_runs": self.total_runs,
            "strategies_in_hall_of_fame": len(self.best_strategies),
            "top_5": self.get_top_strategies(5),
            "best_score_ever": self.best_strategies[0]["score"]
            if self.best_strategies else 0.0,
        }

    def _register_best_strategy(self, optimization_result: Dict[str, Any]):
        """Enregistre une stratégie dans le hall of fame."""
        entry = {
            "strategy": optimization_result["best_strategy"],
            "metrics": optimization_result["best_metrics"],
            "score": optimization_result["best_score"],
            "generations": optimization_result["generations_run"],
            "timestamp": datetime.now().isoformat(),
        }
        self.best_strategies.append(entry)
        if len(self.best_strategies) > 50:
            self.best_strategies.sort(key=lambda x: x["score"], reverse=True)
            self.best_strategies = self.best_strategies[:50]