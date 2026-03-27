"""
micheline/trading/metrics.py

Métriques d'évaluation des stratégies de trading.
Score de 0 à 100. Adapté aux résultats avec capital réel.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("micheline.trading.metrics")


def evaluate_strategy(result: Dict[str, Any]) -> float:
    """
    Évalue une stratégie et retourne un score de 0 à 100.
    Prend en compte le capital réel si disponible.
    """
    if not result or result.get("error"):
        return 0.0

    trades = result.get("trades", 0)
    if trades == 0:
        return 0.0

    profit_pips = result.get("profit", result.get("profit_pips", 0.0))
    profit_money = result.get("profit_money", 0.0)
    winrate = result.get("winrate", 0.0)
    drawdown = result.get("drawdown", 0.0)
    drawdown_pct = result.get("drawdown_pct", 0.0)
    sharpe = result.get("sharpe_ratio", 0.0)
    profit_factor = result.get("profit_factor", 0.0)
    starting_capital = result.get("starting_capital", 10000)
    ending_capital = result.get("ending_capital", 0)

    score = 0.0

    # ── Rendement (max 30 pts) ──
    if starting_capital > 0 and ending_capital > 0:
        pct_return = (ending_capital - starting_capital) / starting_capital * 100
        if pct_return > 0:
            profit_score = min(30.0, pct_return * 0.6)
        else:
            profit_score = max(-10.0, pct_return * 0.3)
    else:
        if profit_pips > 0:
            profit_score = min(30.0, profit_pips / 35.0)
        else:
            profit_score = max(-10.0, profit_pips / 50.0)
    score += profit_score

    # ── Winrate (max 20 pts) ──
    wr_score = min(20.0, winrate * 100.0 * 0.25)
    score += wr_score

    # ── Profit Factor (max 15 pts) ──
    if profit_factor > 1.0:
        pf_score = min(15.0, (profit_factor - 1.0) * 15.0)
    elif profit_factor > 0:
        pf_score = max(-5.0, (profit_factor - 1.0) * 10.0)
    else:
        pf_score = -5.0
    score += pf_score

    # ── Sharpe (max 15 pts) ──
    if sharpe > 0:
        sharpe_score = min(15.0, sharpe * 5.0)
    else:
        sharpe_score = max(-5.0, sharpe * 3.0)
    score += sharpe_score

    # ── Drawdown (pénalité max -15 pts) ──
    if drawdown_pct > 0:
        dd_penalty = min(15.0, drawdown_pct * 0.5)
    elif drawdown > 0:
        dd_penalty = min(15.0, drawdown / 35.0)
    else:
        dd_penalty = 0
    score -= dd_penalty

    # ── Nombre de trades (max 10 pts) ──
    if trades >= 100:
        trades_score = 10.0
    elif trades >= 50:
        trades_score = 7.0
    elif trades >= 20:
        trades_score = 5.0
    elif trades >= 10:
        trades_score = 3.0
    elif trades >= 5:
        trades_score = 1.0
    else:
        trades_score = -5.0
    score += trades_score

    # ── Consistance train/test (max 10 pts) ──
    tts = result.get("train_test_split", {})
    if tts:
        deg = tts.get("degradation_ratio", 1.0)
        cons = tts.get("consistency_score", 1.0)
        if deg >= 0.7 and cons >= 0.7:
            score += 10.0
        elif deg >= 0.4:
            score += 5.0
        elif deg > 0:
            score += 0.0
        else:
            score -= 5.0
    else:
        # Bonus pour gain/perte moyen
        avg_win = abs(result.get("avg_win", 0))
        avg_loss = abs(result.get("avg_loss", 0))
        if avg_loss > 0 and avg_win > 0:
            ratio = avg_win / avg_loss
            if ratio >= 2.0:
                score += 10.0
            elif ratio >= 1.5:
                score += 7.0
            elif ratio >= 1.0:
                score += 4.0

    return round(max(0.0, min(100.0, score)), 1)