"""
micheline/trading/anti_overfit.py

Système anti-overfitting pour la validation des stratégies de trading.
- Split train/test temporel (respecte l'ordre chronologique)
- Validation hors échantillon (OOS)
- Détection d'overfitting par comparaison IS vs OOS
- K-Fold temporel (TimeSeriesSplit)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("micheline.anti_overfit")


class OverfitVerdict(Enum):
    """Verdict de l'analyse anti-overfitting."""
    ROBUST = "robust"
    SUSPECT = "suspect"
    OVERFITTED = "overfitted"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TimePeriod:
    """Représente une période temporelle pour les données."""
    start: datetime
    end: datetime

    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_days": self.duration_days
        }


@dataclass
class SplitResult:
    """Résultat d'un backtest sur un split donné."""
    period: TimePeriod
    split_type: str  # "train", "test", "oos"
    profit: float = 0.0
    drawdown: float = 0.0
    trades: int = 0
    winrate: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    raw_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "period": self.period.to_dict(),
            "split_type": self.split_type,
            "profit": self.profit,
            "drawdown": self.drawdown,
            "trades": self.trades,
            "winrate": self.winrate,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor
        }


@dataclass
class OverfitReport:
    """Rapport complet d'analyse anti-overfitting."""
    strategy_id: str
    verdict: OverfitVerdict
    confidence: float  # 0.0 à 1.0
    train_results: List[SplitResult]
    test_results: List[SplitResult]
    oos_result: Optional[SplitResult]
    degradation_ratio: float  # ratio perf test/train (< 1 = dégradation)
    consistency_score: float  # stabilité entre les folds
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "degradation_ratio": round(self.degradation_ratio, 4),
            "consistency_score": round(self.consistency_score, 4),
            "train_results": [r.to_dict() for r in self.train_results],
            "test_results": [r.to_dict() for r in self.test_results],
            "oos_result": self.oos_result.to_dict() if self.oos_result else None,
            "details": self.details
        }

    @property
    def is_valid(self) -> bool:
        """La stratégie passe-t-elle le test anti-overfitting ?"""
        return self.verdict in (OverfitVerdict.ROBUST, OverfitVerdict.SUSPECT)


class DataSplitter:
    """
    Gère le découpage temporel des données pour éviter l'overfitting.
    
    Principe fondamental : en trading, on ne mélange JAMAIS les données
    temporelles. Le train est TOUJOURS avant le test.
    
    Structure du split :
    |--- TRAIN ---|--- TEST ---|--- GAP ---|--- OOS ---|
    
    Le GAP évite le data leakage entre test et OOS.
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        test_ratio: float = 0.2,
        oos_ratio: float = 0.2,
        gap_days: int = 5,
        min_trades_per_split: int = 30
    ):
        total = train_ratio + test_ratio + oos_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Les ratios doivent sommer à 1.0, got {total}")

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.oos_ratio = oos_ratio
        self.gap_days = gap_days
        self.min_trades_per_split = min_trades_per_split

        logger.info(
            f"DataSplitter initialisé: train={train_ratio}, "
            f"test={test_ratio}, oos={oos_ratio}, gap={gap_days}j"
        )

    def split(
        self, start: datetime, end: datetime
    ) -> Dict[str, TimePeriod]:
        """
        Découpe une période en train/test/OOS avec gap.
        
        Returns:
            {"train": TimePeriod, "test": TimePeriod, "gap": TimePeriod, "oos": TimePeriod}
        """
        total_days = (end - start).days
        gap_days = self.gap_days

        if total_days < 60:
            raise ValueError(
                f"Période trop courte ({total_days}j). Minimum 60 jours requis."
            )

        usable_days = total_days - gap_days
        if usable_days < 45:
            raise ValueError(
                f"Période utilisable trop courte ({usable_days}j) après gap."
            )

        # Calcul des durées proportionnelles sur la partie utilisable
        # Le OOS est pris à la fin, le gap est entre test et OOS
        oos_days = int(usable_days * self.oos_ratio)
        remaining = usable_days - oos_days
        train_days = int(remaining * (self.train_ratio / (self.train_ratio + self.test_ratio)))
        test_days = remaining - train_days

        # Construction des périodes
        train_start = start
        train_end = train_start + timedelta(days=train_days)

        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        gap_start = test_end
        gap_end = gap_start + timedelta(days=gap_days)

        oos_start = gap_end
        oos_end = end

        splits = {
            "train": TimePeriod(train_start, train_end),
            "test": TimePeriod(test_start, test_end),
            "gap": TimePeriod(gap_start, gap_end),
            "oos": TimePeriod(oos_start, oos_end),
        }

        logger.info(f"Split créé sur {total_days}j:")
        for name, period in splits.items():
            logger.info(f"  {name}: {period.start.date()} → {period.end.date()} ({period.duration_days}j)")

        return splits

    def time_series_kfold(
        self, start: datetime, end: datetime, n_folds: int = 5
    ) -> List[Dict[str, TimePeriod]]:
        """
        TimeSeriesSplit : K-Fold adapté aux séries temporelles.
        
        Chaque fold utilise un train croissant et un test fixe qui avance.
        
        Fold 1: |--TRAIN--|--TEST--|
        Fold 2: |----TRAIN----|--TEST--|
        Fold 3: |------TRAIN------|--TEST--|
        ...
        
        Returns:
            Liste de dicts {"train": TimePeriod, "test": TimePeriod}
        """
        total_days = (end - start).days

        if total_days < 30 * n_folds:
            raise ValueError(
                f"Période trop courte ({total_days}j) pour {n_folds} folds. "
                f"Minimum {30 * n_folds}j requis."
            )

        # Chaque fold a un test de taille fixe
        test_size_days = total_days // (n_folds + 1)
        folds = []

        for i in range(n_folds):
            train_end_offset = test_size_days * (i + 1)
            test_end_offset = test_size_days * (i + 2)

            train_period = TimePeriod(
                start,
                start + timedelta(days=train_end_offset)
            )
            test_period = TimePeriod(
                start + timedelta(days=train_end_offset),
                start + timedelta(days=min(test_end_offset, total_days))
            )

            folds.append({
                "train": train_period,
                "test": test_period
            })

            logger.debug(
                f"Fold {i+1}: train={train_period.duration_days}j, "
                f"test={test_period.duration_days}j"
            )

        logger.info(f"TimeSeriesKFold: {n_folds} folds créés sur {total_days}j")
        return folds


class OverfitDetector:
    """
    Détecte l'overfitting en comparant les performances
    in-sample (train) vs out-of-sample (test/OOS).
    """

    # Seuils de détection
    DEGRADATION_THRESHOLD_ROBUST = 0.70    # test >= 70% de train → robust
    DEGRADATION_THRESHOLD_SUSPECT = 0.40   # test >= 40% de train → suspect
    # en dessous de 40% → overfitted

    CONSISTENCY_THRESHOLD = 0.50  # score de consistance minimum
    MIN_OOS_PROFIT_RATIO = 0.30  # OOS doit faire >= 30% du profit train

    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        if custom_thresholds:
            if "degradation_robust" in custom_thresholds:
                self.DEGRADATION_THRESHOLD_ROBUST = custom_thresholds["degradation_robust"]
            if "degradation_suspect" in custom_thresholds:
                self.DEGRADATION_THRESHOLD_SUSPECT = custom_thresholds["degradation_suspect"]
            if "consistency" in custom_thresholds:
                self.CONSISTENCY_THRESHOLD = custom_thresholds["consistency"]
            if "min_oos_profit" in custom_thresholds:
                self.MIN_OOS_PROFIT_RATIO = custom_thresholds["min_oos_profit"]

    def compute_degradation_ratio(
        self,
        train_results: List[SplitResult],
        test_results: List[SplitResult]
    ) -> float:
        """
        Calcule le ratio de dégradation performance test/train.
        
        Un ratio de 1.0 = performances identiques (idéal).
        Un ratio de 0.5 = le test fait 50% du train (suspect).
        Un ratio < 0.3 = probablement overfitted.
        """
        if not train_results or not test_results:
            return 0.0

        # Moyenne des profits normalisés par durée
        avg_train_profit = self._avg_daily_profit(train_results)
        avg_test_profit = self._avg_daily_profit(test_results)

        if avg_train_profit <= 0:
            # Si train négatif, pas de ratio significatif
            return 0.0

        ratio = avg_test_profit / avg_train_profit if avg_train_profit != 0 else 0.0

        # Clamp entre -1 et 2 pour éviter les valeurs aberrantes
        ratio = max(-1.0, min(2.0, ratio))

        logger.info(
            f"Degradation ratio: {ratio:.4f} "
            f"(train_daily={avg_train_profit:.4f}, test_daily={avg_test_profit:.4f})"
        )
        return ratio

    def compute_consistency_score(
        self, test_results: List[SplitResult]
    ) -> float:
        """
        Mesure la consistance des résultats entre les différents folds de test.
        
        Score de 1.0 = tous les folds sont profitables et similaires.
        Score de 0.0 = résultats très incohérents.
        """
        if len(test_results) < 2:
            return 1.0  # Pas assez de folds pour mesurer

        profits = [r.profit for r in test_results]

        # Critère 1 : Proportion de folds profitables
        profitable_ratio = sum(1 for p in profits if p > 0) / len(profits)

        # Critère 2 : Stabilité (coefficient de variation inversé)
        mean_profit = sum(profits) / len(profits)
        if mean_profit == 0:
            stability = 0.0
        else:
            variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
            std_dev = variance ** 0.5
            cv = std_dev / abs(mean_profit) if mean_profit != 0 else float('inf')
            # CV bas = stable, on inverse pour avoir un score
            stability = max(0.0, 1.0 - (cv / 3.0))  # CV de 3 = score 0

        # Critère 3 : Winrate moyen acceptable
        avg_winrate = sum(r.winrate for r in test_results) / len(test_results)
        winrate_score = min(1.0, avg_winrate / 0.45)  # 45% winrate = score 1.0

        # Score composite
        score = (
            profitable_ratio * 0.4 +
            stability * 0.35 +
            winrate_score * 0.25
        )

        logger.info(
            f"Consistency score: {score:.4f} "
            f"(profitable={profitable_ratio:.2f}, stability={stability:.2f}, "
            f"winrate_score={winrate_score:.2f})"
        )
        return score

    def analyze(
        self,
        strategy_id: str,
        train_results: List[SplitResult],
        test_results: List[SplitResult],
        oos_result: Optional[SplitResult] = None
    ) -> OverfitReport:
        """
        Analyse complète anti-overfitting d'une stratégie.
        
        Returns:
            OverfitReport avec verdict, scores et détails.
        """
        logger.info(f"=== Analyse anti-overfitting: {strategy_id} ===")

        # Vérification données suffisantes
        total_trades = sum(r.trades for r in train_results) + sum(r.trades for r in test_results)
        if total_trades < 50:
            logger.warning(f"Données insuffisantes: {total_trades} trades (min 50)")
            return OverfitReport(
                strategy_id=strategy_id,
                verdict=OverfitVerdict.INSUFFICIENT_DATA,
                confidence=0.0,
                train_results=train_results,
                test_results=test_results,
                oos_result=oos_result,
                degradation_ratio=0.0,
                consistency_score=0.0,
                details={"reason": f"Seulement {total_trades} trades, minimum 50 requis"}
            )

        # Calcul des métriques
        degradation = self.compute_degradation_ratio(train_results, test_results)
        consistency = self.compute_consistency_score(test_results)

        # Analyse OOS si disponible
        oos_analysis = {}
        if oos_result:
            oos_analysis = self._analyze_oos(train_results, oos_result)

        # Détermination du verdict
        verdict, confidence = self._determine_verdict(
            degradation, consistency, oos_analysis
        )

        details = {
            "total_trades": total_trades,
            "avg_train_profit": self._avg_daily_profit(train_results),
            "avg_test_profit": self._avg_daily_profit(test_results),
            "oos_analysis": oos_analysis,
            "thresholds_used": {
                "degradation_robust": self.DEGRADATION_THRESHOLD_ROBUST,
                "degradation_suspect": self.DEGRADATION_THRESHOLD_SUSPECT,
                "consistency": self.CONSISTENCY_THRESHOLD,
            }
        }

        report = OverfitReport(
            strategy_id=strategy_id,
            verdict=verdict,
            confidence=confidence,
            train_results=train_results,
            test_results=test_results,
            oos_result=oos_result,
            degradation_ratio=degradation,
            consistency_score=consistency,
            details=details
        )

        logger.info(
            f"Verdict: {verdict.value} (confidence={confidence:.2f}, "
            f"degradation={degradation:.2f}, consistency={consistency:.2f})"
        )

        return report

    def _analyze_oos(
        self,
        train_results: List[SplitResult],
        oos_result: SplitResult
    ) -> dict:
        """Analyse spécifique de la validation OOS."""
        avg_train_daily = self._avg_daily_profit(train_results)
        oos_daily = (
            oos_result.profit / oos_result.period.duration_days
            if oos_result.period.duration_days > 0 else 0.0
        )

        oos_ratio = oos_daily / avg_train_daily if avg_train_daily > 0 else 0.0
        oos_profitable = oos_result.profit > 0

        return {
            "oos_daily_profit": oos_daily,
            "oos_ratio_vs_train": oos_ratio,
            "oos_profitable": oos_profitable,
            "oos_trades": oos_result.trades,
            "oos_winrate": oos_result.winrate,
            "oos_drawdown": oos_result.drawdown,
            "passes_threshold": oos_ratio >= self.MIN_OOS_PROFIT_RATIO
        }

    def _determine_verdict(
        self,
        degradation: float,
        consistency: float,
        oos_analysis: dict
    ) -> Tuple[OverfitVerdict, float]:
        """Détermine le verdict final et le niveau de confiance."""

        scores = []

        # Score basé sur la dégradation
        if degradation >= self.DEGRADATION_THRESHOLD_ROBUST:
            scores.append(1.0)
        elif degradation >= self.DEGRADATION_THRESHOLD_SUSPECT:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score basé sur la consistance
        if consistency >= self.CONSISTENCY_THRESHOLD:
            scores.append(1.0)
        elif consistency >= self.CONSISTENCY_THRESHOLD * 0.6:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score OOS (si disponible, poids important)
        if oos_analysis:
            if oos_analysis.get("passes_threshold", False) and oos_analysis.get("oos_profitable", False):
                scores.append(1.0)
            elif oos_analysis.get("oos_profitable", False):
                scores.append(0.5)
            else:
                scores.append(0.0)

        # Score moyen
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Verdict
        if avg_score >= 0.75:
            verdict = OverfitVerdict.ROBUST
        elif avg_score >= 0.40:
            verdict = OverfitVerdict.SUSPECT
        else:
            verdict = OverfitVerdict.OVERFITTED

        # La confiance est basée sur la quantité de données et la clarté du verdict
        confidence = min(1.0, abs(avg_score - 0.5) * 2 + 0.3)

        return verdict, confidence

    @staticmethod
    def _avg_daily_profit(results: List[SplitResult]) -> float:
        """Calcule le profit journalier moyen sur une liste de résultats."""
        if not results:
            return 0.0

        total_profit = sum(r.profit for r in results)
        total_days = sum(r.period.duration_days for r in results)

        if total_days == 0:
            return 0.0

        return total_profit / total_days


class AntiOverfitValidator:
    """
    Classe principale qui orchestre toute la validation anti-overfitting.
    
    Usage:
        validator = AntiOverfitValidator(backtest_runner=my_backtest_func)
        report = validator.validate_strategy(strategy, start, end)
        if report.is_valid:
            # Stratégie acceptée
    """

    def __init__(
        self,
        backtest_runner=None,
        splitter: Optional[DataSplitter] = None,
        detector: Optional[OverfitDetector] = None,
        n_folds: int = 5
    ):
        """
        Args:
            backtest_runner: Callable(strategy_config, start, end) -> dict avec
                             keys: profit, drawdown, trades, winrate, sharpe_ratio, profit_factor
            splitter: DataSplitter personnalisé (optionnel)
            detector: OverfitDetector personnalisé (optionnel)
            n_folds: Nombre de folds pour la cross-validation temporelle
        """
        self.backtest_runner = backtest_runner
        self.splitter = splitter or DataSplitter()
        self.detector = detector or OverfitDetector()
        self.n_folds = n_folds

        logger.info(f"AntiOverfitValidator initialisé avec {n_folds} folds")

    def validate_strategy(
        self,
        strategy_config: dict,
        data_start: datetime,
        data_end: datetime,
        strategy_id: str = "unnamed"
    ) -> OverfitReport:
        """
        Valide une stratégie complète contre l'overfitting.
        
        Processus:
        1. Split principal train/test/OOS
        2. Cross-validation temporelle sur la partie train+test
        3. Backtest final sur OOS (jamais vu)
        4. Analyse et verdict
        """
        if not self.backtest_runner:
            raise RuntimeError("backtest_runner non défini. Impossible de valider.")

        logger.info(f"=== Validation anti-overfit: {strategy_id} ===")
        logger.info(f"Période: {data_start.date()} → {data_end.date()}")

        # Étape 1 : Split principal pour isoler l'OOS
        main_split = self.splitter.split(data_start, data_end)

        # Étape 2 : Cross-validation temporelle sur train+test (hors OOS)
        cv_start = main_split["train"].start
        cv_end = main_split["test"].end

        try:
            folds = self.splitter.time_series_kfold(cv_start, cv_end, self.n_folds)
        except ValueError as e:
            logger.warning(f"Impossible de faire {self.n_folds} folds: {e}")
            # Fallback : split simple
            folds = [{"train": main_split["train"], "test": main_split["test"]}]

        # Étape 3 : Exécuter les backtests sur chaque fold
        train_results = []
        test_results = []

        for i, fold in enumerate(folds):
            logger.info(f"--- Fold {i+1}/{len(folds)} ---")

            # Backtest train
            train_bt = self._run_backtest_safe(
                strategy_config,
                fold["train"],
                f"fold_{i+1}_train"
            )
            if train_bt:
                train_results.append(train_bt)

            # Backtest test
            test_bt = self._run_backtest_safe(
                strategy_config,
                fold["test"],
                f"fold_{i+1}_test"
            )
            if test_bt:
                test_results.append(test_bt)

        # Étape 4 : Backtest OOS (la partie jamais vue)
        oos_result = self._run_backtest_safe(
            strategy_config,
            main_split["oos"],
            "oos"
        )

        # Étape 5 : Analyse
        report = self.detector.analyze(
            strategy_id=strategy_id,
            train_results=train_results,
            test_results=test_results,
            oos_result=oos_result
        )

        return report

    def quick_validate(
        self,
        train_metrics: dict,
        test_metrics: dict,
        strategy_id: str = "unnamed"
    ) -> OverfitReport:
        """
        Validation rapide sans exécuter de backtests.
        Utile quand on a déjà les résultats train/test.
        
        Args:
            train_metrics: {"profit": x, "drawdown": x, "trades": x, "winrate": x, ...}
            test_metrics: idem
        """
        now = datetime.now()

        train_result = SplitResult(
            period=TimePeriod(now - timedelta(days=200), now - timedelta(days=100)),
            split_type="train",
            profit=train_metrics.get("profit", 0),
            drawdown=train_metrics.get("drawdown", 0),
            trades=train_metrics.get("trades", 0),
            winrate=train_metrics.get("winrate", 0),
            sharpe_ratio=train_metrics.get("sharpe_ratio", 0),
            profit_factor=train_metrics.get("profit_factor", 0)
        )

        test_result = SplitResult(
            period=TimePeriod(now - timedelta(days=100), now),
            split_type="test",
            profit=test_metrics.get("profit", 0),
            drawdown=test_metrics.get("drawdown", 0),
            trades=test_metrics.get("trades", 0),
            winrate=test_metrics.get("winrate", 0),
            sharpe_ratio=test_metrics.get("sharpe_ratio", 0),
            profit_factor=test_metrics.get("profit_factor", 0)
        )

        return self.detector.analyze(
            strategy_id=strategy_id,
            train_results=[train_result],
            test_results=[test_result],
            oos_result=None
        )

    def _run_backtest_safe(
        self,
        strategy_config: dict,
        period: TimePeriod,
        label: str
    ) -> Optional[SplitResult]:
        """Exécute un backtest avec gestion d'erreur."""
        try:
            logger.info(
                f"Backtest [{label}]: {period.start.date()} → {period.end.date()}"
            )

            result = self.backtest_runner(
                strategy_config,
                period.start,
                period.end
            )

            if not isinstance(result, dict):
                logger.error(f"Backtest [{label}] a retourné un type invalide: {type(result)}")
                return None

            split_type = "train" if "train" in label else ("oos" if "oos" in label else "test")

            return SplitResult(
                period=period,
                split_type=split_type,
                profit=result.get("profit", 0.0),
                drawdown=result.get("drawdown", 0.0),
                trades=result.get("trades", 0),
                winrate=result.get("winrate", 0.0),
                sharpe_ratio=result.get("sharpe_ratio", 0.0),
                profit_factor=result.get("profit_factor", 0.0),
                raw_results=result
            )

        except Exception as e:
            logger.error(f"Erreur backtest [{label}]: {e}")
            return None