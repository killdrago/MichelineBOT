"""
micheline/trading/optimizer.py

Optimisation des stratégies de trading avec validation anti-overfitting (Phase 7).
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from trading.metrics import evaluate_strategy, compute_robustness_score
from trading.anti_overfit import (
    AntiOverfitValidator,
    DataSplitter,
    OverfitDetector,
    OverfitVerdict
)
from trading.walk_forward import WalkForwardAnalyzer

logger = logging.getLogger("micheline.optimizer")


class StrategyOptimizer:
    """
    Optimiseur de stratégies avec validation anti-overfitting intégrée.
    
    Le cycle d'optimisation :
    1. Générer une stratégie candidate
    2. Évaluer sur données d'entraînement
    3. Valider avec anti-overfitting (Phase 7)
    4. Garder ou rejeter
    5. Améliorer les survivantes
    """

    def __init__(
        self,
        backtest_runner: Callable = None,
        strategy_generator: Callable = None,
        strategy_improver: Callable = None,
        max_iterations: int = 50,
        min_robustness_score: float = 40.0,
        enable_walk_forward: bool = True,
        enable_anti_overfit: bool = True
    ):
        """
        Args:
            backtest_runner: Callable(config, start, end) -> dict résultat backtest
            strategy_generator: Callable() -> dict config stratégie
            strategy_improver: Callable(config, score) -> dict config améliorée
            max_iterations: Nombre max d'itérations d'optimisation
            min_robustness_score: Score minimum de robustesse pour accepter
            enable_walk_forward: Activer le walk-forward analysis
            enable_anti_overfit: Activer la validation anti-overfitting
        """
        self.backtest_runner = backtest_runner
        self.strategy_generator = strategy_generator
        self.strategy_improver = strategy_improver
        self.max_iterations = max_iterations
        self.min_robustness_score = min_robustness_score
        self.enable_walk_forward = enable_walk_forward
        self.enable_anti_overfit = enable_anti_overfit

        # Composants anti-overfitting
        self.anti_overfit_validator = AntiOverfitValidator(
            backtest_runner=backtest_runner,
            n_folds=5
        )
        self.walk_forward_analyzer = WalkForwardAnalyzer(
            backtest_runner=backtest_runner,
            optimization_days=120,
            trading_days=30,
            step_days=30
        )

        # Historique
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_strategies: List[Dict[str, Any]] = []

        logger.info(
            f"StrategyOptimizer: max_iter={max_iterations}, "
            f"min_robustness={min_robustness_score}, "
            f"walk_forward={'ON' if enable_walk_forward else 'OFF'}, "
            f"anti_overfit={'ON' if enable_anti_overfit else 'OFF'}"
        )

    def optimize(
        self,
        data_start: datetime,
        data_end: datetime,
        n_candidates: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lance le processus d'optimisation complet.
        
        Returns:
            Liste des meilleures stratégies validées, triées par score.
        """
        if not self.backtest_runner or not self.strategy_generator:
            raise RuntimeError("backtest_runner et strategy_generator requis")

        logger.info(f"=== Optimisation: {n_candidates} candidats, {self.max_iterations} itérations ===")

        validated_strategies = []

        for candidate_idx in range(n_candidates):
            logger.info(f"--- Candidat {candidate_idx + 1}/{n_candidates} ---")

            # Étape 1 : Générer
            strategy = self.strategy_generator()
            strategy_id = f"strat_{candidate_idx}"

            # Étape 2 : Itérer et améliorer
            best_score = 0.0
            best_config = strategy

            for iteration in range(self.max_iterations):
                # Backtest simple (rapide)
                try:
                    result = self.backtest_runner(strategy, data_start, data_end)
                    score = evaluate_strategy(result)
                except Exception as e:
                    logger.error(f"Erreur backtest itération {iteration}: {e}")
                    break

                if score > best_score:
                    best_score = score
                    best_config = strategy.copy()

                # Améliorer
                if self.strategy_improver:
                    strategy = self.strategy_improver(strategy, score)

                # Early stopping si score trop bas après plusieurs itérations
                if iteration > 10 and best_score < 20.0:
                    logger.info(f"Early stop: score trop bas ({best_score:.1f}) après {iteration} itérations")
                    break

            logger.info(f"Candidat {strategy_id}: meilleur score brut = {best_score:.1f}")

            # Étape 3 : Validation anti-overfitting
            if self.enable_anti_overfit and best_score > 15.0:
                validation_result = self._validate_candidate(
                    best_config, data_start, data_end, strategy_id
                )

                if validation_result["accepted"]:
                    validated_strategies.append({
                        "strategy_id": strategy_id,
                        "config": best_config,
                        "raw_score": best_score,
                        "robustness_score": validation_result["robustness_score"],
                        "adjusted_score": validation_result["adjusted_score"],
                        "overfit_verdict": validation_result["overfit_verdict"],
                        "walk_forward_verdict": validation_result.get("wf_verdict"),
                        "details": validation_result
                    })
                    logger.info(f"✅ Candidat {strategy_id} ACCEPTÉ (adjusted={validation_result['adjusted_score']:.1f})")
                else:
                    logger.info(f"❌ Candidat {strategy_id} REJETÉ ({validation_result.get('rejection_reason', 'unknown')})")
            elif best_score > 15.0:
                # Sans anti-overfit, accepter directement
                validated_strategies.append({
                    "strategy_id": strategy_id,
                    "config": best_config,
                    "raw_score": best_score,
                    "robustness_score": best_score,  # Pas de validation
                    "adjusted_score": best_score,
                    "overfit_verdict": "not_tested",
                    "details": {}
                })

            self.optimization_history.append({
                "strategy_id": strategy_id,
                "best_score": best_score,
                "accepted": any(s["strategy_id"] == strategy_id for s in validated_strategies)
            })

        # Trier par score ajusté
        validated_strategies.sort(key=lambda x: x["adjusted_score"], reverse=True)

        self.best_strategies = validated_strategies

        logger.info(
            f"=== Optimisation terminée: {len(validated_strategies)}/{n_candidates} "
            f"stratégies validées ==="
        )

        return validated_strategies

    def _validate_candidate(
        self,
        strategy_config: dict,
        data_start: datetime,
        data_end: datetime,
        strategy_id: str
    ) -> Dict[str, Any]:
        """
        Validation complète d'un candidat avec anti-overfitting et walk-forward.
        
        Returns:
            Dict avec accepted, scores, verdicts, etc.
        """
        result = {
            "accepted": False,
            "robustness_score": 0.0,
            "adjusted_score": 0.0,
            "overfit_verdict": "not_tested",
            "rejection_reason": None
        }

        # Test 1 : Anti-overfitting (split train/test/OOS)
        try:
            overfit_report = self.anti_overfit_validator.validate_strategy(
                strategy_config, data_start, data_end, strategy_id
            )

            result["overfit_verdict"] = overfit_report.verdict.value
            result["overfit_report"] = overfit_report.to_dict()

            if overfit_report.verdict == OverfitVerdict.OVERFITTED:
                result["rejection_reason"] = f"Overfitted (degradation={overfit_report.degradation_ratio:.2f})"
                return result

            # Calculer le score de robustesse
            train_scores = [evaluate_strategy(r.raw_results) for r in overfit_report.train_results if r.raw_results]
            test_scores = [evaluate_strategy(r.raw_results) for r in overfit_report.test_results if r.raw_results]

            avg_train_score = sum(train_scores) / len(train_scores) if train_scores else 0
            avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0

            oos_score = None
            if overfit_report.oos_result and overfit_report.oos_result.raw_results:
                oos_score = evaluate_strategy(overfit_report.oos_result.raw_results)

            robustness = compute_robustness_score(
                train_score=avg_train_score,
                test_score=avg_test_score,
                oos_score=oos_score,
                overfit_report=overfit_report.to_dict()
            )

            result["robustness_score"] = robustness["robustness_score"]
            result["adjusted_score"] = robustness["adjusted_score"]

        except Exception as e:
            logger.error(f"Erreur validation anti-overfit: {e}")
            result["overfit_verdict"] = "error"

        # Test 2 : Walk-Forward (si activé)
        if self.enable_walk_forward:
            try:
                wf_report = self.walk_forward_analyzer.run(
                    strategy_config, data_start, data_end, strategy_id
                )

                result["wf_verdict"] = wf_report.verdict.value
                result["wf_report"] = wf_report.to_dict()

                if wf_report.verdict == OverfitVerdict.OVERFITTED:
                    result["rejection_reason"] = f"Walk-forward failed (WFE={wf_report.wfe:.2%})"
                    return result

                # Ajuster le score avec les résultats WF
                if wf_report.verdict == OverfitVerdict.ROBUST:
                    result["adjusted_score"] *= 1.10  # Bonus 10%
                elif wf_report.verdict == OverfitVerdict.SUSPECT:
                    result["adjusted_score"] *= 0.90  # Malus 10%

            except Exception as e:
                logger.error(f"Erreur walk-forward: {e}")
                result["wf_verdict"] = "error"

        # Décision finale
        if result["robustness_score"] >= self.min_robustness_score:
            result["accepted"] = True
        else:
            result["rejection_reason"] = (
                f"Robustness trop faible ({result['robustness_score']:.1f} "
                f"< {self.min_robustness_score})"
            )

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'optimisation."""
        return {
            "total_candidates": len(self.optimization_history),
            "accepted": sum(1 for h in self.optimization_history if h["accepted"]),
            "rejected": sum(1 for h in self.optimization_history if not h["accepted"]),
            "best_strategy": self.best_strategies[0] if self.best_strategies else None,
            "all_validated": [
                {
                    "id": s["strategy_id"],
                    "adjusted_score": s["adjusted_score"],
                    "robustness": s["robustness_score"],
                    "overfit_verdict": s["overfit_verdict"]
                }
                for s in self.best_strategies
            ]
        }