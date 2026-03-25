"""
micheline/tests/test_anti_overfit.py

Tests unitaires pour la Phase 7 — Anti-Overfitting.
"""

import unittest
from datetime import datetime, timedelta
from trading.anti_overfit import (
    DataSplitter,
    OverfitDetector,
    AntiOverfitValidator,
    OverfitVerdict,
    TimePeriod,
    SplitResult
)
from trading.monte_carlo import MonteCarloSimulator
from trading.walk_forward import WalkForwardAnalyzer
from trading.metrics import (
    evaluate_strategy,
    compute_robustness_score,
    compute_sharpe_ratio,
    compute_profit_factor
)


class TestDataSplitter(unittest.TestCase):
    """Tests pour DataSplitter."""

    def setUp(self):
        self.splitter = DataSplitter(
            train_ratio=0.6, test_ratio=0.2, oos_ratio=0.2, gap_days=5
        )
        self.start = datetime(2023, 1, 1)
        self.end = datetime(2024, 1, 1)  # 365 jours

    def test_split_basic(self):
        """Test split basique."""
        splits = self.splitter.split(self.start, self.end)
        self.assertIn("train", splits)
        self.assertIn("test", splits)
        self.assertIn("gap", splits)
        self.assertIn("oos", splits)

    def test_split_order(self):
        """Les périodes doivent être ordonnées chronologiquement."""
        splits = self.splitter.split(self.start, self.end)
        self.assertLessEqual(splits["train"].end, splits["test"].start)
        self.assertLessEqual(splits["test"].end, splits["gap"].start)
        self.assertLessEqual(splits["gap"].end, splits["oos"].start)

    def test_split_too_short(self):
        """Période trop courte doit lever une erreur."""
        short_end = self.start + timedelta(days=30)
        with self.assertRaises(ValueError):
            self.splitter.split(self.start, short_end)

    def test_kfold(self):
        """Test TimeSeriesKFold."""
        folds = self.splitter.time_series_kfold(self.start, self.end, n_folds=5)
        self.assertEqual(len(folds), 5)
        # Chaque fold a train et test
        for fold in folds:
            self.assertIn("train", fold)
            self.assertIn("test", fold)
            # Train avant test
            self.assertLessEqual(fold["train"].end, fold["test"].start)

    def test_kfold_expanding_train(self):
        """Le train doit être croissant dans le KFold temporel."""
        folds = self.splitter.time_series_kfold(self.start, self.end, n_folds=4)
        train_sizes = [f["train"].duration_days for f in folds]
        for i in range(1, len(train_sizes)):
            self.assertGreater(train_sizes[i], train_sizes[i-1])


class TestOverfitDetector(unittest.TestCase):
    """Tests pour OverfitDetector."""

    def setUp(self):
        self.detector = OverfitDetector()
        self.now = datetime.now()

    def _make_result(self, profit, trades, winrate, days_ago_start, days_ago_end, split_type):
        return SplitResult(
            period=TimePeriod(
                self.now - timedelta(days=days_ago_start),
                self.now - timedelta(days=days_ago_end)
            ),
            split_type=split_type,
            profit=profit,
            drawdown=abs(profit) * 0.3,
            trades=trades,
            winrate=winrate,
            sharpe_ratio=1.5 if profit > 0 else -0.5
        )

    def test_robust_strategy(self):
        """Stratégie avec bonnes perf train et test → ROBUST."""
        train = [self._make_result(1000, 50, 0.55, 200, 100, "train")]
        test = [self._make_result(800, 40, 0.52, 100, 50, "test")]

        report = self.detector.analyze("test_robust", train, test)
        self.assertEqual(report.verdict, OverfitVerdict.ROBUST)

    def test_overfitted_strategy(self):
        """Stratégie excellente en train, nulle en test → OVERFITTED."""
        train = [self._make_result(5000, 100, 0.70, 200, 100, "train")]
        test = [self._make_result(-500, 40, 0.35, 100, 50, "test")]

        report = self.detector.analyze("test_overfit", train, test)
        self.assertEqual(report.verdict, OverfitVerdict.OVERFITTED)

    def test_insufficient_data(self):
        """Trop peu de trades → INSUFFICIENT_DATA."""
        train = [self._make_result(100, 10, 0.60, 200, 100, "train")]
        test = [self._make_result(50, 5, 0.50, 100, 50, "test")]

        report = self.detector.analyze("test_insufficient", train, test)
        self.assertEqual(report.verdict, OverfitVerdict.INSUFFICIENT_DATA)

    def test_report_is_valid(self):
        """Test de la propriété is_valid."""
        train = [self._make_result(1000, 50, 0.55, 200, 100, "train")]
        test = [self._make_result(800, 40, 0.52, 100, 50, "test")]
        report = self.detector.analyze("test", train, test)
        self.assertTrue(report.is_valid)


class TestMonteCarloSimulator(unittest.TestCase):
    """Tests pour MonteCarloSimulator."""

    def setUp(self):
        self.mc = MonteCarloSimulator(n_simulations=500, seed=42)

    def test_profitable_strategy(self):
        """Stratégie clairement profitable → statistiquement significative."""
        # Beaucoup de petits gains, quelques pertes
        trades = [10.0] * 80 + [-5.0] * 20  # 80% win, avg gain > avg loss
        result = self.mc.run_bootstrap_test(trades)
        self.assertTrue(result.is_statistically_significant)
        self.assertGreater(result.confidence_level, 0.90)

    def test_random_strategy(self):
        """Stratégie aléatoire → probablement non significative."""
        import random
        random.seed(42)
        trades = [random.uniform(-10, 10) for _ in range(100)]
        result = self.mc.run_bootstrap_test(trades)
        # On ne peut pas garantir le verdict, mais on vérifie la structure
        self.assertIsNotNone(result.percentile_5)
        self.assertIsNotNone(result.probability_of_loss)

    def test_noise_test(self):
        """Test de bruit."""
        trades = [10.0] * 60 + [-5.0] * 40
        result = self.mc.run_noise_test(trades, noise_pct=0.15)
        self.assertEqual(result.n_simulations, 500)

    def test_full_analysis(self):
        """Test analyse complète."""
        trades = [15.0] * 70 + [-8.0] * 30
        results = self.mc.run_full_analysis(trades)
        self.assertIn("permutation", results)
        self.assertIn("bootstrap", results)
        self.assertIn("noise", results)


class TestMetrics(unittest.TestCase):
    """Tests pour les nouvelles métriques Phase 7."""

    def test_sharpe_ratio(self):
        """Test calcul Sharpe ratio."""
        # Trades consistants → Sharpe élevé
        trades = [10.0] * 100
        sharpe = compute_sharpe_ratio(trades)
        # Std dev = 0, donc Sharpe = 0 (division par 0 gérée)
        self.assertEqual(sharpe, 0.0)

        # Trades avec variance
        trades = [10.0, -2.0, 8.0, -1.0, 12.0, -3.0] * 20
        sharpe = compute_sharpe_ratio(trades)
        self.assertGreater(sharpe, 0)

    def test_profit_factor(self):
        """Test calcul profit factor."""
        trades = [10.0, 10.0, 10.0, -5.0, -5.0]
        pf = compute_profit_factor(trades)
        self.assertEqual(pf, 3.0)  # 30 / 10

    def test_robustness_score_good(self):
        """Bon score de robustesse."""
        result = compute_robustness_score(
            train_score=70.0,
            test_score=60.0,
            oos_score=55.0
        )
        self.assertGreater(result["robustness_score"], 50.0)
        self.assertGreater(result["adjusted_score"], 40.0)

    def test_robustness_score_overfit(self):
        """Score de robustesse pour stratégie overfittée."""
        result = compute_robustness_score(
            train_score=90.0,
            test_score=10.0
        )
        self.assertLess(result["adjusted_score"], 30.0)
        self.assertGreater(result["penalty"], 0)


class TestAntiOverfitValidator(unittest.TestCase):
    """Tests pour AntiOverfitValidator."""

    def _mock_backtest_good(self, config, start, end):
        """Simule un backtest de bonne stratégie."""
        return {
            "profit": 500.0,
            "drawdown": 5.0,
            "trades": 50,
            "winrate": 0.55,
            "sharpe_ratio": 1.2,
            "profit_factor": 1.8
        }

    def _mock_backtest_overfit(self, config, start, end):
        """Simule un backtest de stratégie overfittée (dépend de la période)."""
        # Fonctionne bien sur le début, mal sur la fin
        mid = start + (end - start) / 2
        if end < mid + timedelta(days=30):
            return {
                "profit": 2000.0, "drawdown": 3.0, "trades": 80,
                "winrate": 0.70, "sharpe_ratio": 2.5, "profit_factor": 3.0
            }
        else:
            return {
                "profit": -200.0, "drawdown": 25.0, "trades": 30,
                "winrate": 0.35, "sharpe_ratio": -0.5, "profit_factor": 0.6
            }

    def test_quick_validate_good(self):
        """Validation rapide d'une bonne stratégie."""
        validator = AntiOverfitValidator()
        report = validator.quick_validate(
            train_metrics={"profit": 1000, "drawdown": 5, "trades": 60, "winrate": 0.55},
            test_metrics={"profit": 800, "drawdown": 7, "trades": 40, "winrate": 0.52},
            strategy_id="quick_good"
        )
        self.assertTrue(report.is_valid)

    def test_quick_validate_bad(self):
        """Validation rapide d'une stratégie overfittée."""
        validator = AntiOverfitValidator()
        report = validator.quick_validate(
            train_metrics={"profit": 5000, "drawdown": 2, "trades": 100, "winrate": 0.75},
            test_metrics={"profit": -500, "drawdown": 30, "trades": 50, "winrate": 0.30},
            strategy_id="quick_bad"
        )
        self.assertFalse(report.is_valid)

    def test_full_validate_with_mock(self):
        """Validation complète avec backtest mocké."""
        validator = AntiOverfitValidator(
            backtest_runner=self._mock_backtest_good,
            n_folds=3
        )
        report = validator.validate_strategy(
            strategy_config={"type": "test"},
            data_start=datetime(2023, 1, 1),
            data_end=datetime(2024, 1, 1),
            strategy_id="full_test"
        )
        self.assertIsNotNone(report)
        self.assertIn(report.verdict, list(OverfitVerdict))


class TestWalkForward(unittest.TestCase):
    """Tests pour WalkForwardAnalyzer."""

    def _mock_backtest(self, config, start, end):
        return {
            "profit": 300.0,
            "drawdown": 8.0,
            "trades": 25,
            "winrate": 0.52,
            "sharpe_ratio": 1.0,
            "profit_factor": 1.5
        }

    def test_generate_windows(self):
        """Test génération des fenêtres."""
        wfa = WalkForwardAnalyzer(
            optimization_days=90,
            trading_days=30,
            step_days=30
        )
        windows = wfa.generate_windows(
            datetime(2023, 1, 1),
            datetime(2024, 1, 1)
        )
        self.assertGreater(len(windows), 0)
        # Vérifier l'ordre
        for w in windows:
            self.assertLessEqual(
                w.optimization_period.end,
                w.trading_period.start
            )

    def test_run_walk_forward(self):
        """Test exécution complète du walk-forward."""
        wfa = WalkForwardAnalyzer(
            backtest_runner=self._mock_backtest,
            optimization_days=60,
            trading_days=20,
            step_days=20
        )
        report = wfa.run(
            strategy_config={"type": "test"},
            data_start=datetime(2023, 1, 1),
            data_end=datetime(2024, 1, 1),
            strategy_id="wf_test"
        )
        self.assertIsNotNone(report)
        self.assertGreater(report.overall_trades, 0)


if __name__ == "__main__":
    unittest.main()