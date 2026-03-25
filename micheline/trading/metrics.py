"""
micheline/trading/metrics.py

Métriques d'évaluation des stratégies de trading.
Inclut le scoring anti-overfitting (Phase 7).
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("micheline.metrics")


def evaluate_strategy(result: dict) -> float:
    """
    Évalue un résultat de backtest et retourne un score composite.
    Score entre 0.0 (mauvais) et 100.0 (excellent).

    Args:
        result: dict avec keys: profit, drawdown, trades, winrate,
                optionnellement: sharpe_ratio, profit_factor

    Returns:
        Score composite float
    """
    profit = result.get("profit", 0)
    drawdown = result.get("drawdown", 0)
    trades = result.get("trades", 0)
    winrate = result.get("winrate", 0)
    sharpe = result.get("sharpe_ratio", 0)
    profit_factor = result.get("profit_factor", 0)

    # Composante profit (0-30 points)
    profit_score = min(30.0, max(0.0, profit / 100.0))

    # Composante drawdown (0-25 points) — moins c'est mieux
    if drawdown <= 0:
        dd_score = 25.0
    elif drawdown < 5:
        dd_score = 20.0
    elif drawdown < 10:
        dd_score = 15.0
    elif drawdown < 20:
        dd_score = 10.0
    elif drawdown < 30:
        dd_score = 5.0
    else:
        dd_score = 0.0

    # Composante winrate (0-20 points)
    winrate_score = min(20.0, winrate * 20.0 / 0.6)  # 60% winrate = 20 pts

    # Composante nombre de trades (0-10 points) — éviter trop peu
    if trades >= 100:
        trades_score = 10.0
    elif trades >= 50:
        trades_score = 7.0
    elif trades >= 30:
        trades_score = 5.0
    elif trades >= 10:
        trades_score = 3.0
    else:
        trades_score = 1.0

    # Composante Sharpe (0-10 points)
    sharpe_score = min(10.0, max(0.0, sharpe * 5.0))

    # Composante profit factor (0-5 points)
    pf_score = min(5.0, max(0.0, (profit_factor - 1.0) * 5.0)) if profit_factor > 0 else 0.0

    total = profit_score + dd_score + winrate_score + trades_score + sharpe_score + pf_score

    logger.debug(
        f"Score breakdown: profit={profit_score:.1f}, dd={dd_score:.1f}, "
        f"winrate={winrate_score:.1f}, trades={trades_score:.1f}, "
        f"sharpe={sharpe_score:.1f}, pf={pf_score:.1f} → total={total:.1f}"
    )

    return round(total, 2)


def compute_robustness_score(
    train_score: float,
    test_score: float,
    oos_score: Optional[float] = None,
    overfit_report: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Calcule un score de robustesse combinant performance et anti-overfitting.
    
    NOUVEAU — Phase 7
    
    Ce score est utilisé par l'optimizer pour décider si une stratégie
    vaut la peine d'être gardée.
    
    Args:
        train_score: Score sur données d'entraînement
        test_score: Score sur données de test
        oos_score: Score sur données OOS (optionnel)
        overfit_report: Rapport d'OverfitDetector.to_dict() (optionnel)
    
    Returns:
        Dict avec:
            - robustness_score: float (0-100)
            - adjusted_score: float (score final ajusté)
            - penalty: float (pénalité appliquée)
            - breakdown: dict des composantes
    """
    # Ratio de dégradation train → test
    if train_score > 0:
        degradation = test_score / train_score
    else:
        degradation = 0.0

    # Pénalité pour overfitting
    # Plus la dégradation est forte, plus la pénalité est élevée
    if degradation >= 0.80:
        overfit_penalty = 0.0  # Très bon : pas de pénalité
    elif degradation >= 0.60:
        overfit_penalty = 0.15  # Acceptable
    elif degradation >= 0.40:
        overfit_penalty = 0.35  # Suspect
    else:
        overfit_penalty = 0.60  # Overfitted

    # Score de base = moyenne pondérée test (principal) et train (secondaire)
    base_score = test_score * 0.7 + train_score * 0.3

    # Bonus/malus OOS
    oos_adjustment = 0.0
    if oos_score is not None:
        if oos_score >= test_score * 0.7:
            oos_adjustment = 5.0  # Bonus : OOS confirme le test
        elif oos_score >= test_score * 0.4:
            oos_adjustment = 0.0  # Neutre
        else:
            oos_adjustment = -10.0  # Malus : OOS ne confirme pas

    # Bonus/malus du rapport anti-overfitting
    report_adjustment = 0.0
    if overfit_report:
        verdict = overfit_report.get("verdict", "")
        if verdict == "robust":
            report_adjustment = 10.0
        elif verdict == "suspect":
            report_adjustment = -5.0
        elif verdict == "overfitted":
            report_adjustment = -20.0

    # Score final
    penalty = overfit_penalty * base_score
    adjusted_score = base_score - penalty + oos_adjustment + report_adjustment
    adjusted_score = max(0.0, min(100.0, adjusted_score))

    # Score de robustesse (indépendant de la performance)
    robustness = (degradation * 50) + (min(1.0, degradation) * 50)
    if oos_score is not None and train_score > 0:
        oos_ratio = min(1.0, oos_score / train_score)
        robustness = robustness * 0.6 + oos_ratio * 100 * 0.4
    robustness = max(0.0, min(100.0, robustness))

    result = {
        "robustness_score": round(robustness, 2),
        "adjusted_score": round(adjusted_score, 2),
        "penalty": round(penalty, 2),
        "breakdown": {
            "train_score": round(train_score, 2),
            "test_score": round(test_score, 2),
            "oos_score": round(oos_score, 2) if oos_score else None,
            "degradation_ratio": round(degradation, 4),
            "overfit_penalty": round(overfit_penalty, 4),
            "oos_adjustment": round(oos_adjustment, 2),
            "report_adjustment": round(report_adjustment, 2),
            "base_score": round(base_score, 2)
        }
    }

    logger.info(
        f"Robustness: score={robustness:.1f}, adjusted={adjusted_score:.1f}, "
        f"penalty={penalty:.1f}, degradation={degradation:.2f}"
    )

    return result


def compute_sharpe_ratio(
    trade_results: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calcule le ratio de Sharpe à partir des P&L des trades.
    
    NOUVEAU — Phase 7 (utilitaire pour les évaluations)
    
    Args:
        trade_results: Liste des P&L de chaque trade
        risk_free_rate: Taux sans risque annualisé (défaut 0)
        periods_per_year: Nombre de périodes par an (252 jours de trading)
    
    Returns:
        Sharpe ratio annualisé
    """
    if len(trade_results) < 2:
        return 0.0

    n = len(trade_results)
    mean_return = sum(trade_results) / n
    variance = sum((r - mean_return) ** 2 for r in trade_results) / (n - 1)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return 0.0

    daily_sharpe = (mean_return - risk_free_rate / periods_per_year) / std_dev
    annualized_sharpe = daily_sharpe * (periods_per_year ** 0.5)

    return round(annualized_sharpe, 4)


def compute_profit_factor(trade_results: List[float]) -> float:
    """
    Calcule le profit factor (gains bruts / pertes brutes).
    
    NOUVEAU — Phase 7
    
    Args:
        trade_results: Liste des P&L de chaque trade
    
    Returns:
        Profit factor (> 1.0 = profitable)
    """
    gross_profit = sum(t for t in trade_results if t > 0)
    gross_loss = abs(sum(t for t in trade_results if t < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return round(gross_profit / gross_loss, 4)