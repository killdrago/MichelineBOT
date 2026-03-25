"""
micheline/core/evaluator.py

Évaluateur central du système Micheline.
Intègre la validation anti-overfitting (Phase 7) dans le pipeline d'évaluation.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("micheline.evaluator")


class Evaluator:
    """
    Évalue les résultats des actions de l'agent.
    En Phase 7, intègre la validation anti-overfitting pour les stratégies de trading.
    """

    def __init__(self):
        self.evaluation_history: list = []
        self._anti_overfit_validator = None
        self._monte_carlo_simulator = None

        logger.info("Evaluator initialisé")

    def _get_anti_overfit_validator(self):
        """Lazy loading du validateur anti-overfitting."""
        if self._anti_overfit_validator is None:
            try:
                from trading.anti_overfit import AntiOverfitValidator
                self._anti_overfit_validator = AntiOverfitValidator()
            except ImportError:
                logger.warning("Module anti_overfit non disponible")
        return self._anti_overfit_validator

    def _get_monte_carlo_simulator(self):
        """Lazy loading du simulateur Monte Carlo."""
        if self._monte_carlo_simulator is None:
            try:
                from trading.monte_carlo import MonteCarloSimulator
                self._monte_carlo_simulator = MonteCarloSimulator(n_simulations=500)
            except ImportError:
                logger.warning("Module monte_carlo non disponible")
        return self._monte_carlo_simulator

    def evaluate(self, result: dict, context: Optional[dict] = None) -> dict:
        """
        Évalue un résultat d'action.
        
        Args:
            result: Résultat brut de l'action exécutée
            context: Contexte optionnel (type d'action, objectif, etc.)
        
        Returns:
            Dict avec score, verdict, recommandation
        """
        context = context or {}
        action_type = context.get("type", "unknown")

        if action_type == "backtest" or action_type == "trading":
            evaluation = self._evaluate_trading_result(result, context)
        elif action_type == "code_modification":
            evaluation = self._evaluate_code_result(result, context)
        else:
            evaluation = self._evaluate_generic(result, context)

        # Stocker dans l'historique
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "evaluation": evaluation
        })

        return evaluation

    def _evaluate_trading_result(self, result: dict, context: dict) -> dict:
        """
        Évalue un résultat de trading avec validation anti-overfitting.
        
        C'est ici que la Phase 7 est intégrée dans le pipeline principal.
        """
        from trading.metrics import evaluate_strategy, compute_robustness_score

        # Score de base
        base_score = evaluate_strategy(result)

        evaluation = {
            "base_score": base_score,
            "verdict": "pending",
            "recommendation": "none",
            "anti_overfit": None,
            "monte_carlo": None
        }

        # Validation anti-overfitting si données suffisantes
        trade_results = result.get("trade_results", [])  # Liste des P&L individuels

        if trade_results and len(trade_results) >= 30:
            mc_sim = self._get_monte_carlo_simulator()
            if mc_sim:
                try:
                    mc_results = mc_sim.run_full_analysis(trade_results)
                    evaluation["monte_carlo"] = {
                        name: r.to_dict() for name, r in mc_results.items()
                    }

                    # Vérifier si statistiquement significatif
                    all_significant = all(
                        r.is_statistically_significant for r in mc_results.values()
                    )
                    if not all_significant:
                        evaluation["verdict"] = "statistically_questionable"
                        evaluation["recommendation"] = "reject_or_investigate"
                        logger.warning("Stratégie statistiquement non significative (Monte Carlo)")

                except Exception as e:
                    logger.error(f"Erreur Monte Carlo: {e}")

        # Score de robustesse si on a train/test séparés
        train_score = context.get("train_score")
        test_score = context.get("test_score")

        if train_score is not None and test_score is not None:
            robustness = compute_robustness_score(
                train_score=train_score,
                test_score=test_score,
                oos_score=context.get("oos_score"),
                overfit_report=context.get("overfit_report")
            )
            evaluation["robustness"] = robustness
            evaluation["adjusted_score"] = robustness["adjusted_score"]
        else:
            evaluation["adjusted_score"] = base_score

        # Verdict final
        if evaluation["verdict"] == "pending":
            adj = evaluation["adjusted_score"]
            if adj >= 60:
                evaluation["verdict"] = "excellent"
                evaluation["recommendation"] = "deploy"
            elif adj >= 40:
                evaluation["verdict"] = "acceptable"
                evaluation["recommendation"] = "continue_optimization"
            elif adj >= 20:
                evaluation["verdict"] = "mediocre"
                evaluation["recommendation"] = "major_revision"
            else:
                evaluation["verdict"] = "poor"
                evaluation["recommendation"] = "abandon"

        logger.info(
            f"Trading evaluation: base={base_score:.1f}, "
            f"adjusted={evaluation['adjusted_score']:.1f}, "
            f"verdict={evaluation['verdict']}"
        )

        return evaluation

    def _evaluate_code_result(self, result: dict, context: dict) -> dict:
        """Évalue un résultat de modification de code."""
        success = result.get("success", False)
        tests_passed = result.get("tests_passed", False)
        errors = result.get("errors", [])

        score = 0.0
        if success:
            score += 50.0
        if tests_passed:
            score += 50.0
        score -= len(errors) * 10.0
        score = max(0.0, score)

        return {
            "base_score": score,
            "adjusted_score": score,
            "verdict": "success" if score >= 80 else "needs_review",
            "recommendation": "apply" if score >= 80 else "rollback",
            "errors": errors
        }

    def _evaluate_generic(self, result: dict, context: dict) -> dict:
        """Évaluation générique pour les actions non catégorisées."""
        success = result.get("success", False)
        return {
            "base_score": 100.0 if success else 0.0,
            "adjusted_score": 100.0 if success else 0.0,
            "verdict": "success" if success else "failure",
            "recommendation": "continue" if success else "retry"
        }

    def should_abandon_strategy(self, strategy_id: str) -> bool:
        """
        Détermine si une stratégie devrait être abandonnée
        basé sur l'historique des évaluations.
        """
        strategy_evals = [
            e for e in self.evaluation_history
            if e.get("evaluation", {}).get("strategy_id") == strategy_id
        ]

        if len(strategy_evals) < 3:
            return False

        recent = strategy_evals[-3:]
        avg_score = sum(
            e.get("evaluation", {}).get("adjusted_score", 0)
            for e in recent
        ) / 3

        if avg_score < 15.0:
            logger.info(f"Stratégie {strategy_id} recommandée à l'abandon (avg={avg_score:.1f})")
            return True

        return False