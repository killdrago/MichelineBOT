"""
micheline/trading/monte_carlo.py

Simulation Monte Carlo pour la validation de stratégies de trading.
Teste si les résultats sont statistiquement significatifs ou dus au hasard.

Méthodes :
1. Permutation des trades (shuffle)
2. Perturbation des résultats (bruit)
3. Resampling bootstrap
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("micheline.monte_carlo")


@dataclass
class MonteCarloResult:
    """Résultat d'une simulation Monte Carlo."""
    n_simulations: int
    original_profit: float
    original_drawdown: float
    simulated_profits: List[float]
    simulated_drawdowns: List[float]
    percentile_5: float    # 5ème percentile des profits simulés
    percentile_50: float   # Médiane
    percentile_95: float   # 95ème percentile
    probability_of_loss: float  # P(profit < 0) sur les simulations
    worst_drawdown_95: float  # 95ème percentile des drawdowns (pire cas réaliste)
    confidence_level: float  # Niveau de confiance que la strat est réellement profitable
    is_statistically_significant: bool

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "original_profit": round(self.original_profit, 2),
            "original_drawdown": round(self.original_drawdown, 2),
            "percentile_5": round(self.percentile_5, 2),
            "percentile_50": round(self.percentile_50, 2),
            "percentile_95": round(self.percentile_95, 2),
            "probability_of_loss": round(self.probability_of_loss, 4),
            "worst_drawdown_95": round(self.worst_drawdown_95, 2),
            "confidence_level": round(self.confidence_level, 4),
            "is_statistically_significant": self.is_statistically_significant
        }


class MonteCarloSimulator:
    """
    Simulations Monte Carlo pour valider la robustesse d'une stratégie.
    
    Principe : on perturbe les trades de la stratégie de nombreuses façons,
    et on vérifie que la stratégie reste profitable dans la majorité des cas.
    Si elle ne l'est qu'avec l'ordre exact des trades originaux, c'est suspect.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_threshold: float = 0.95,
        seed: Optional[int] = None
    ):
        self.n_simulations = n_simulations
        self.confidence_threshold = confidence_threshold

        if seed is not None:
            random.seed(seed)

        logger.info(
            f"MonteCarloSimulator: {n_simulations} simulations, "
            f"confidence={confidence_threshold}"
        )

    def run_permutation_test(
        self, trade_results: List[float]
    ) -> MonteCarloResult:
        """
        Test de permutation : mélange l'ordre des trades.
        
        Si la stratégie est robuste, l'ordre des trades ne devrait pas
        impacter drastiquement le résultat final (le profit total reste le même,
        mais le drawdown change).
        
        Args:
            trade_results: Liste des P&L de chaque trade
        """
        if len(trade_results) < 10:
            logger.warning(f"Seulement {len(trade_results)} trades. Résultats peu fiables.")

        original_profit = sum(trade_results)
        original_equity = self._compute_equity_curve(trade_results)
        original_drawdown = self._compute_max_drawdown(original_equity)

        simulated_profits = []
        simulated_drawdowns = []

        for _ in range(self.n_simulations):
            shuffled = trade_results.copy()
            random.shuffle(shuffled)

            equity = self._compute_equity_curve(shuffled)
            dd = self._compute_max_drawdown(equity)

            simulated_profits.append(sum(shuffled))  # Même total (permutation)
            simulated_drawdowns.append(dd)

        return self._build_result(
            original_profit, original_drawdown,
            simulated_profits, simulated_drawdowns
        )

    def run_bootstrap_test(
        self, trade_results: List[float]
    ) -> MonteCarloResult:
        """
        Test bootstrap : rééchantillonne les trades avec remplacement.
        
        Certains trades seront comptés plusieurs fois, d'autres ignorés.
        Cela simule "d'autres versions possibles" de l'historique.
        
        Args:
            trade_results: Liste des P&L de chaque trade
        """
        if len(trade_results) < 10:
            logger.warning(f"Seulement {len(trade_results)} trades. Résultats peu fiables.")

        original_profit = sum(trade_results)
        original_equity = self._compute_equity_curve(trade_results)
        original_drawdown = self._compute_max_drawdown(original_equity)

        n = len(trade_results)
        simulated_profits = []
        simulated_drawdowns = []

        for _ in range(self.n_simulations):
            # Rééchantillonnage avec remplacement
            resampled = [random.choice(trade_results) for _ in range(n)]

            equity = self._compute_equity_curve(resampled)
            dd = self._compute_max_drawdown(equity)

            simulated_profits.append(sum(resampled))
            simulated_drawdowns.append(dd)

        return self._build_result(
            original_profit, original_drawdown,
            simulated_profits, simulated_drawdowns
        )

    def run_noise_test(
        self, trade_results: List[float], noise_pct: float = 0.10
    ) -> MonteCarloResult:
        """
        Test de bruit : ajoute du bruit aléatoire à chaque trade.
        
        Simule l'incertitude : en réalité, le slippage, le spread,
        les conditions de marché font que les résultats ne sont jamais exacts.
        
        Args:
            trade_results: Liste des P&L de chaque trade
            noise_pct: Pourcentage de bruit (0.10 = ±10%)
        """
        if len(trade_results) < 10:
            logger.warning(f"Seulement {len(trade_results)} trades. Résultats peu fiables.")

        original_profit = sum(trade_results)
        original_equity = self._compute_equity_curve(trade_results)
        original_drawdown = self._compute_max_drawdown(original_equity)

        simulated_profits = []
        simulated_drawdowns = []

        for _ in range(self.n_simulations):
            noisy = []
            for trade in trade_results:
                noise = trade * random.uniform(-noise_pct, noise_pct)
                noisy.append(trade + noise)

            equity = self._compute_equity_curve(noisy)
            dd = self._compute_max_drawdown(equity)

            simulated_profits.append(sum(noisy))
            simulated_drawdowns.append(dd)

        return self._build_result(
            original_profit, original_drawdown,
            simulated_profits, simulated_drawdowns
        )

    def run_full_analysis(
        self, trade_results: List[float]
    ) -> Dict[str, MonteCarloResult]:
        """
        Exécute les trois tests et retourne un rapport complet.
        """
        logger.info(f"=== Monte Carlo Full Analysis ({len(trade_results)} trades) ===")

        results = {}

        logger.info("Running permutation test...")
        results["permutation"] = self.run_permutation_test(trade_results)

        logger.info("Running bootstrap test...")
        results["bootstrap"] = self.run_bootstrap_test(trade_results)

        logger.info("Running noise test...")
        results["noise"] = self.run_noise_test(trade_results)

        # Verdict global
        all_significant = all(r.is_statistically_significant for r in results.values())
        avg_confidence = sum(r.confidence_level for r in results.values()) / len(results)

        logger.info(
            f"Monte Carlo verdict: all_significant={all_significant}, "
            f"avg_confidence={avg_confidence:.2%}"
        )

        return results

    def _build_result(
        self,
        original_profit: float,
        original_drawdown: float,
        simulated_profits: List[float],
        simulated_drawdowns: List[float]
    ) -> MonteCarloResult:
        """Construit le résultat Monte Carlo à partir des simulations."""

        sorted_profits = sorted(simulated_profits)
        sorted_drawdowns = sorted(simulated_drawdowns)

        n = len(sorted_profits)

        percentile_5 = sorted_profits[int(n * 0.05)]
        percentile_50 = sorted_profits[int(n * 0.50)]
        percentile_95 = sorted_profits[int(n * 0.95)]

        probability_of_loss = sum(1 for p in simulated_profits if p < 0) / n
        worst_drawdown_95 = sorted_drawdowns[int(n * 0.95)]

        confidence_level = 1.0 - probability_of_loss

        is_significant = (
            confidence_level >= self.confidence_threshold
            and percentile_5 > 0  # Même le pire scénario réaliste est profitable
        )

        return MonteCarloResult(
            n_simulations=n,
            original_profit=original_profit,
            original_drawdown=original_drawdown,
            simulated_profits=simulated_profits,
            simulated_drawdowns=simulated_drawdowns,
            percentile_5=percentile_5,
            percentile_50=percentile_50,
            percentile_95=percentile_95,
            probability_of_loss=probability_of_loss,
            worst_drawdown_95=worst_drawdown_95,
            confidence_level=confidence_level,
            is_statistically_significant=is_significant
        )

    @staticmethod
    def _compute_equity_curve(trades: List[float]) -> List[float]:
        """Calcule la courbe d'équité à partir des P&L."""
        equity = [0.0]
        for trade in trades:
            equity.append(equity[-1] + trade)
        return equity

    @staticmethod
    def _compute_max_drawdown(equity: List[float]) -> float:
        """Calcule le drawdown maximum sur une courbe d'équité."""
        if not equity:
            return 0.0

        peak = equity[0]
        max_dd = 0.0

        for value in equity:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd