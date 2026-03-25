# micheline/trading/metrics.py
"""
Métriques d'évaluation de stratégies de trading.
Calcule un score composite pour comparer les stratégies entre elles.
"""

import math
from typing import Dict, Any, Optional


def evaluate_result(backtest_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Prend le résultat brut d'un backtest MT5 et calcule
    toutes les métriques dérivées.

    Args:
        backtest_result: dict retourné par mt5_tools.run_backtest()
            Attendu : {
                "profit": float,
                "drawdown": float,
                "trades": int,
                "winrate": float,
                # optionnels :
                "gross_profit": float,
                "gross_loss": float,
                "initial_deposit": float,
                "max_consecutive_losses": int,
                "avg_trade_duration_minutes": float,
                "equity_curve": list[float],  (optionnel)
            }

    Returns:
        dict de toutes les métriques calculées
    """
    profit = backtest_result.get("profit", 0.0)
    drawdown = backtest_result.get("drawdown", 0.0)
    trades = backtest_result.get("trades", 0)
    winrate = backtest_result.get("winrate", 0.0)
    initial_deposit = backtest_result.get("initial_deposit", 10000.0)
    gross_profit = backtest_result.get("gross_profit", max(0.0, profit))
    gross_loss = backtest_result.get("gross_loss", min(0.0, profit) if profit < 0 else 0.0)
    equity_curve = backtest_result.get("equity_curve", None)

    metrics = {
        "profit": profit,
        "drawdown": drawdown,
        "trades": trades,
        "winrate": winrate,
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "return_pct": _return_pct(profit, initial_deposit),
        "risk_reward_ratio": _risk_reward_ratio(profit, drawdown),
        "sharpe_ratio": _sharpe_ratio(equity_curve, profit, trades),
        "recovery_factor": _recovery_factor(profit, drawdown),
        "avg_trade_profit": _avg_trade_profit(profit, trades),
        "expectancy": _expectancy(winrate, gross_profit, gross_loss, trades),
    }

    metrics["composite_score"] = compute_score(metrics)

    return metrics


def compute_score(metrics: Dict[str, float]) -> float:
    """
    Calcule un score composite unique pour classer les stratégies.

    Pondérations :
        - profit normalisé      : 25%
        - drawdown (pénalité)   : 20%
        - sharpe ratio          : 20%
        - winrate               : 15%
        - profit factor         : 10%
        - nombre de trades      : 10%

    Le score est entre 0 et 100.
    """
    profit = metrics.get("profit", 0.0)
    drawdown = metrics.get("drawdown", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    winrate = metrics.get("winrate", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)
    trades = metrics.get("trades", 0)

    # ── Composante Profit (0-100) ──
    # On normalise : 1000$ de profit = score 50, 5000$ = 100
    profit_score = min(100.0, max(0.0, (profit / 5000.0) * 100.0))

    # ── Composante Drawdown (0-100, inversé) ──
    # 0% drawdown = 100, 50% drawdown = 0
    drawdown_score = max(0.0, 100.0 - (drawdown * 2.0))

    # ── Composante Sharpe (0-100) ──
    # Sharpe de 2 = 66, Sharpe de 3 = 100
    sharpe_score = min(100.0, max(0.0, (sharpe / 3.0) * 100.0))

    # ── Composante Winrate (0-100) ──
    winrate_score = min(100.0, max(0.0, winrate))

    # ── Composante Profit Factor (0-100) ──
    # PF de 1.5 = 50, PF de 3 = 100
    pf_score = min(100.0, max(0.0, (profit_factor / 3.0) * 100.0))

    # ── Composante Trades (0-100) ──
    # On veut assez de trades pour la signification statistique
    # 30 trades = 50, 100 trades = 100
    trades_score = min(100.0, max(0.0, (trades / 100.0) * 100.0))

    # ── Score composite pondéré ──
    composite = (
        profit_score * 0.25
        + drawdown_score * 0.20
        + sharpe_score * 0.20
        + winrate_score * 0.15
        + pf_score * 0.10
        + trades_score * 0.10
    )

    # ── Pénalités ──
    # Trop peu de trades = stratégie non fiable
    if trades < 10:
        composite *= 0.3
    elif trades < 30:
        composite *= 0.7

    # Drawdown excessif = danger
    if drawdown > 40:
        composite *= 0.5
    elif drawdown > 25:
        composite *= 0.8

    # Profit négatif = plancher
    if profit < 0:
        composite = min(composite, 15.0)

    return round(composite, 2)


def compare_strategies(
    metrics_a: Dict[str, float], metrics_b: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare deux stratégies et retourne un résumé.

    Args:
        metrics_a: métriques stratégie A
        metrics_b: métriques stratégie B

    Returns:
        dict avec le gagnant et les différences
    """
    score_a = metrics_a.get("composite_score", compute_score(metrics_a))
    score_b = metrics_b.get("composite_score", compute_score(metrics_b))

    differences = {}
    common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
    for key in common_keys:
        if isinstance(metrics_a[key], (int, float)) and isinstance(
            metrics_b[key], (int, float)
        ):
            differences[key] = {
                "a": metrics_a[key],
                "b": metrics_b[key],
                "diff": round(metrics_a[key] - metrics_b[key], 4),
            }

    return {
        "winner": "A" if score_a >= score_b else "B",
        "score_a": score_a,
        "score_b": score_b,
        "score_diff": round(score_a - score_b, 2),
        "details": differences,
    }


def is_strategy_viable(metrics: Dict[str, float]) -> bool:
    """
    Détermine si une stratégie mérite d'être conservée / optimisée.

    Critères minimaux :
        - profit > 0
        - drawdown < 40%
        - trades >= 10
        - winrate >= 30%
        - composite_score >= 25
    """
    if metrics.get("profit", 0) <= 0:
        return False
    if metrics.get("drawdown", 100) > 40:
        return False
    if metrics.get("trades", 0) < 10:
        return False
    if metrics.get("winrate", 0) < 30:
        return False
    if metrics.get("composite_score", 0) < 25:
        return False
    return True


# ──────────────────────────────────────────────
# Fonctions internes de calcul
# ──────────────────────────────────────────────


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Ratio profit brut / perte brute."""
    abs_loss = abs(gross_loss)
    if abs_loss == 0:
        return 10.0 if gross_profit > 0 else 0.0
    return round(gross_profit / abs_loss, 4)


def _return_pct(profit: float, initial_deposit: float) -> float:
    """Rendement en pourcentage."""
    if initial_deposit == 0:
        return 0.0
    return round((profit / initial_deposit) * 100.0, 2)


def _risk_reward_ratio(profit: float, drawdown: float) -> float:
    """Ratio rendement / risque (drawdown)."""
    if drawdown == 0:
        return 10.0 if profit > 0 else 0.0
    return round(profit / drawdown, 4)


def _sharpe_ratio(
    equity_curve: Optional[list], profit: float, trades: int
) -> float:
    """
    Sharpe ratio simplifié.
    Si equity_curve est fourni, calcul sur les rendements.
    Sinon, estimation grossière.
    """
    if equity_curve and len(equity_curve) > 2:
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] != 0:
                r = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(r)

        if not returns:
            return 0.0

        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(variance) if variance > 0 else 0.001

        # Annualisation approximative (252 jours)
        sharpe = (mean_r / std_r) * math.sqrt(252)
        return round(max(-5.0, min(5.0, sharpe)), 4)

    # Estimation si pas d'equity curve
    if trades == 0:
        return 0.0
    avg_profit_per_trade = profit / trades
    estimated_sharpe = avg_profit_per_trade / max(abs(profit) * 0.1, 1.0)
    return round(max(-5.0, min(5.0, estimated_sharpe)), 4)


def _recovery_factor(profit: float, drawdown: float) -> float:
    """Facteur de récupération = profit net / drawdown max."""
    if drawdown == 0:
        return 10.0 if profit > 0 else 0.0
    return round(profit / drawdown, 4)


def _avg_trade_profit(profit: float, trades: int) -> float:
    """Profit moyen par trade."""
    if trades == 0:
        return 0.0
    return round(profit / trades, 4)


def _expectancy(
    winrate: float, gross_profit: float, gross_loss: float, trades: int
) -> float:
    """
    Espérance mathématique par trade.
    E = (winrate * avg_win) - ((1-winrate) * avg_loss)
    """
    if trades == 0:
        return 0.0

    win_rate_decimal = winrate / 100.0
    winning_trades = max(1, int(trades * win_rate_decimal))
    losing_trades = max(1, trades - winning_trades)

    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = abs(gross_loss) / losing_trades if losing_trades > 0 else 0

    expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
    return round(expectancy, 4)