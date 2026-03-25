"""
micheline/trading/walk_forward.py

Walk-Forward Analysis (WFA) pour la validation de stratégies.
Simule le processus réel : optimiser sur le passé, trader sur le futur,
avancer dans le temps, recommencer.

C'est le test le plus réaliste pour une stratégie de trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable

from trading.anti_overfit import TimePeriod, SplitResult, OverfitVerdict

logger = logging.getLogger("micheline.walk_forward")


@dataclass
class WalkForwardWindow:
    """Une fenêtre du walk-forward."""
    window_id: int
    optimization_period: TimePeriod  # Période d'optimisation (in-sample)
    trading_period: TimePeriod       # Période de trading (out-of-sample)
    optimized_params: dict = field(default_factory=dict)
    optimization_result: Optional[SplitResult] = None
    trading_result: Optional[SplitResult] = None

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "optimization_period": self.optimization_period.to_dict(),
            "trading_period": self.trading_period.to_dict(),
            "optimized_params": self.optimized_params,
            "optimization_result": self.optimization_result.to_dict() if self.optimization_result else None,
            "trading_result": self.trading_result.to_dict() if self.trading_result else None
        }


@dataclass
class WalkForwardReport:
    """Rapport complet du Walk-Forward Analysis."""
    strategy_id: str
    windows: List[WalkForwardWindow]
    overall_profit: float
    overall_trades: int
    overall_winrate: float
    wfe: float  # Walk-Forward Efficiency
    verdict: OverfitVerdict
    consistency_ratio: float  # % de fenêtres profitables
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "windows": [w.to_dict() for w in self.windows],
            "overall_profit": round(self.overall_profit, 2),
            "overall_trades": self.overall_trades,
            "overall_winrate": round(self.overall_winrate, 4),
            "walk_forward_efficiency": round(self.wfe, 4),
            "verdict": self.verdict.value,
            "consistency_ratio": round(self.consistency_ratio, 4),
            "details": self.details
        }

    @property
    def is_valid(self) -> bool:
        return self.verdict in (OverfitVerdict.ROBUST, OverfitVerdict.SUSPECT)


class WalkForwardAnalyzer:
    """
    Implémente le Walk-Forward Analysis.
    
    Principe :
    1. Diviser les données en N fenêtres glissantes
    2. Pour chaque fenêtre :
       a. Optimiser les paramètres sur la partie "optimization"
       b. Tester avec ces paramètres sur la partie "trading" (jamais vue)
    3. Assembler tous les résultats "trading" = performance réaliste
    
    |--OPT1--|--TRADE1--|
         |--OPT2--|--TRADE2--|
              |--OPT3--|--TRADE3--|
    """

    def __init__(
        self,
        backtest_runner: Callable = None,
        optimizer_func: Callable = None,
        optimization_days: int = 120,
        trading_days: int = 30,
        step_days: int = 30,
        min_trades_per_window: int = 15
    ):
        """
        Args:
            backtest_runner: Callable(config, start, end) -> dict
            optimizer_func: Callable(config, start, end) -> optimized_config
                           Si None, utilise la config telle quelle
            optimization_days: Taille de la fenêtre d'optimisation
            trading_days: Taille de la fenêtre de trading
            step_days: Pas d'avancement entre les fenêtres
            min_trades_per_window: Minimum de trades pour une fenêtre valide
        """
        self.backtest_runner = backtest_runner
        self.optimizer_func = optimizer_func
        self.optimization_days = optimization_days
        self.trading_days = trading_days
        self.step_days = step_days
        self.min_trades_per_window = min_trades_per_window

        logger.info(
            f"WalkForwardAnalyzer: opt={optimization_days}j, "
            f"trade={trading_days}j, step={step_days}j"
        )

    def generate_windows(
        self, start: datetime, end: datetime
    ) -> List[WalkForwardWindow]:
        """Génère les fenêtres de walk-forward."""
        windows = []
        window_id = 0
        current = start

        while True:
            opt_start = current
            opt_end = opt_start + timedelta(days=self.optimization_days)
            trade_start = opt_end
            trade_end = trade_start + timedelta(days=self.trading_days)

            if trade_end > end:
                break

            window = WalkForwardWindow(
                window_id=window_id,
                optimization_period=TimePeriod(opt_start, opt_end),
                trading_period=TimePeriod(trade_start, trade_end)
            )
            windows.append(window)

            current += timedelta(days=self.step_days)
            window_id += 1

        logger.info(f"Walk-forward: {len(windows)} fenêtres générées")

        if len(windows) < 3:
            logger.warning(
                f"Seulement {len(windows)} fenêtres. "
                f"Résultats peu fiables (min recommandé: 5)"
            )

        return windows

    def run(
        self,
        strategy_config: dict,
        data_start: datetime,
        data_end: datetime,
        strategy_id: str = "unnamed"
    ) -> WalkForwardReport:
        """
        Exécute l'analyse walk-forward complète.
        
        Returns:
            WalkForwardReport avec tous les résultats et le verdict.
        """
        if not self.backtest_runner:
            raise RuntimeError("backtest_runner non défini")

        logger.info(f"=== Walk-Forward Analysis: {strategy_id} ===")

        windows = self.generate_windows(data_start, data_end)
        if not windows:
            raise ValueError("Aucune fenêtre générée. Période trop courte?")

        for window in windows:
            logger.info(f"--- Window {window.window_id + 1}/{len(windows)} ---")

            # Étape 1 : Optimisation (si optimizer disponible)
            if self.optimizer_func:
                try:
                    optimized_config = self.optimizer_func(
                        strategy_config,
                        window.optimization_period.start,
                        window.optimization_period.end
                    )
                    window.optimized_params = optimized_config
                except Exception as e:
                    logger.error(f"Erreur optimisation window {window.window_id}: {e}")
                    window.optimized_params = strategy_config
            else:
                window.optimized_params = strategy_config

            # Backtest sur la période d'optimisation (pour référence)
            try:
                opt_result = self.backtest_runner(
                    window.optimized_params,
                    window.optimization_period.start,
                    window.optimization_period.end
                )
                window.optimization_result = SplitResult(
                    period=window.optimization_period,
                    split_type="train",
                    profit=opt_result.get("profit", 0),
                    drawdown=opt_result.get("drawdown", 0),
                    trades=opt_result.get("trades", 0),
                    winrate=opt_result.get("winrate", 0),
                    sharpe_ratio=opt_result.get("sharpe_ratio", 0),
                    profit_factor=opt_result.get("profit_factor", 0),
                    raw_results=opt_result
                )
            except Exception as e:
                logger.error(f"Erreur backtest optimization window {window.window_id}: {e}")

            # Étape 2 : Trading (out-of-sample)
            try:
                trade_result = self.backtest_runner(
                    window.optimized_params,
                    window.trading_period.start,
                    window.trading_period.end
                )
                window.trading_result = SplitResult(
                    period=window.trading_period,
                    split_type="test",
                    profit=trade_result.get("profit", 0),
                    drawdown=trade_result.get("drawdown", 0),
                    trades=trade_result.get("trades", 0),
                    winrate=trade_result.get("winrate", 0),
                    sharpe_ratio=trade_result.get("sharpe_ratio", 0),
                    profit_factor=trade_result.get("profit_factor", 0),
                    raw_results=trade_result
                )
            except Exception as e:
                logger.error(f"Erreur backtest trading window {window.window_id}: {e}")

        # Analyse des résultats
        report = self._build_report(strategy_id, windows)
        return report

    def _build_report(
        self, strategy_id: str, windows: List[WalkForwardWindow]
    ) -> WalkForwardReport:
        """Construit le rapport final à partir des fenêtres."""

        # Filtrer les fenêtres avec résultats de trading
        valid_windows = [w for w in windows if w.trading_result is not None]

        if not valid_windows:
            return WalkForwardReport(
                strategy_id=strategy_id,
                windows=windows,
                overall_profit=0.0,
                overall_trades=0,
                overall_winrate=0.0,
                wfe=0.0,
                verdict=OverfitVerdict.INSUFFICIENT_DATA,
                consistency_ratio=0.0,
                details={"error": "Aucune fenêtre avec résultats valides"}
            )

        # Métriques globales sur les périodes de trading uniquement
        overall_profit = sum(w.trading_result.profit for w in valid_windows)
        overall_trades = sum(w.trading_result.trades for w in valid_windows)

        total_winning = sum(
            w.trading_result.trades * w.trading_result.winrate
            for w in valid_windows
        )
        overall_winrate = total_winning / overall_trades if overall_trades > 0 else 0.0

        # Walk-Forward Efficiency (WFE)
        # = profit OOS / profit IS × 100
        # Un WFE > 50% est généralement acceptable
        total_opt_profit = sum(
            w.optimization_result.profit
            for w in valid_windows
            if w.optimization_result
        )
        wfe = overall_profit / total_opt_profit if total_opt_profit > 0 else 0.0

        # Consistance : % de fenêtres profitables
        profitable_windows = sum(
            1 for w in valid_windows if w.trading_result.profit > 0
        )
        consistency_ratio = profitable_windows / len(valid_windows)

        # Verdict
        verdict = self._determine_verdict(wfe, consistency_ratio, valid_windows)

        report = WalkForwardReport(
            strategy_id=strategy_id,
            windows=windows,
            overall_profit=overall_profit,
            overall_trades=overall_trades,
            overall_winrate=overall_winrate,
            wfe=wfe,
            verdict=verdict,
            consistency_ratio=consistency_ratio,
            details={
                "total_windows": len(windows),
                "valid_windows": len(valid_windows),
                "profitable_windows": profitable_windows,
                "total_opt_profit": total_opt_profit,
                "wfe_percentage": f"{wfe * 100:.1f}%"
            }
        )

        logger.info(
            f"Walk-Forward Report: profit={overall_profit:.2f}, "
            f"WFE={wfe*100:.1f}%, consistency={consistency_ratio*100:.0f}%, "
            f"verdict={verdict.value}"
        )

        return report

    @staticmethod
    def _determine_verdict(
        wfe: float,
        consistency: float,
        windows: List[WalkForwardWindow]
    ) -> OverfitVerdict:
        """Détermine le verdict du walk-forward."""

        if len(windows) < 3:
            return OverfitVerdict.INSUFFICIENT_DATA

        # Critères
        wfe_ok = wfe >= 0.50  # WFE >= 50%
        consistency_ok = consistency >= 0.60  # >= 60% fenêtres profitables
        wfe_acceptable = wfe >= 0.30

        if wfe_ok and consistency_ok:
            return OverfitVerdict.ROBUST
        elif wfe_acceptable and consistency >= 0.45:
            return OverfitVerdict.SUSPECT
        else:
            return OverfitVerdict.OVERFITTED