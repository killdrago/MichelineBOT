# micheline/trading/formatter.py
"""
Formateur de résultats trading.
Transforme les dicts bruts en texte lisible avec détails complets.
"""

from typing import Dict, Any, List


def format_trading_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Formate un résultat trading selon le type d'outil."""
    formatters = {
        "trading_quick_test": _format_quick_test,
        "trading_search": _format_search,
        "trading_evaluate": _format_evaluate,
        "trading_improve": _format_improve,
        "trading_report": _format_report,
        "trading_top_strategies": _format_top_strategies,
    }

    formatter = formatters.get(tool_name)
    if formatter:
        try:
            return formatter(result)
        except Exception as e:
            return f"📊 Résultat trading (erreur formatage: {e})\n{str(result)[:500]}"

    return str(result)


def _format_strategy_details(strategy: dict) -> list:
    """Formate les détails d'une stratégie en lignes de texte."""
    lines = []

    if not strategy:
        return ["      (pas de détails disponibles)"]

    # Indicateurs
    indicators = strategy.get("indicators", [])
    if indicators:
        ind_parts = []
        for ind in indicators:
            name = ind.get("name", "?")
            params = ind.get("params", {})
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                ind_parts.append(f"{name}({param_str})")
            else:
                ind_parts.append(name)
        lines.append(f"      📐 Indicateurs : {', '.join(ind_parts)}")

    # Logique d'entrée
    entry = strategy.get("entry_logic", {})
    if entry:
        entry_type = entry.get("type", "?")
        direction = entry.get("direction", "both")
        conditions = entry.get("conditions", [])

        lines.append(f"      🎯 Entrée : {entry_type} ({direction})")

        for cond in conditions[:3]:
            cond_type = cond.get("type", "?")
            if cond_type == "crossover":
                fast = cond.get("fast_indicator", "?")
                slow = cond.get("slow_indicator", "?")
                lines.append(f"         → Croisement {fast} / {slow}")
            elif cond_type in ("threshold_above", "threshold_below"):
                ind = cond.get("indicator", "?")
                thresh = cond.get("threshold", "?")
                direction_word = "au-dessus de" if "above" in cond_type else "en-dessous de"
                lines.append(f"         → {ind} {direction_word} {thresh}")
            elif cond_type == "breakout":
                lookback = cond.get("lookback_period", "?")
                lines.append(f"         → Cassure sur {lookback} périodes")
            elif cond_type == "mean_reversion":
                ind = cond.get("indicator", "?")
                dev = cond.get("deviation_threshold", "?")
                lines.append(f"         → Retour à la moyenne {ind} (dév: {dev})")
            elif cond_type == "divergence":
                ind = cond.get("indicator", "?")
                lines.append(f"         → Divergence {ind}")
            elif cond_type == "trend_filter":
                ind = cond.get("indicator", "?")
                lines.append(f"         → Filtre tendance {ind}")

    # Logique de sortie
    exit_l = strategy.get("exit_logic", {})
    if exit_l:
        exit_type = exit_l.get("type", "?")
        exit_params = exit_l.get("params", {})

        if exit_type == "fixed_tp_sl":
            tp = exit_params.get("take_profit_pips", "?")
            sl = exit_params.get("stop_loss_pips", "?")
            lines.append(f"      🚪 Sortie : TP {tp} pips / SL {sl} pips")
        elif exit_type == "trailing_stop":
            trail = exit_params.get("trail_distance_pips", "?")
            lines.append(f"      🚪 Sortie : Trailing stop {trail} pips")
        elif exit_type == "atr_based":
            tp_mult = exit_params.get("tp_atr_multiplier", "?")
            sl_mult = exit_params.get("sl_atr_multiplier", "?")
            lines.append(f"      🚪 Sortie : ATR (TP x{tp_mult} / SL x{sl_mult})")
        elif exit_type == "time_based":
            bars = exit_params.get("max_bars_in_trade", "?")
            lines.append(f"      🚪 Sortie : Après {bars} bougies max")
        elif exit_type == "indicator_exit":
            ind = exit_params.get("exit_indicator", "?")
            lines.append(f"      🚪 Sortie : Signal {ind}")
        elif exit_type == "opposite_signal":
            lines.append(f"      🚪 Sortie : Signal inverse")
        else:
            lines.append(f"      🚪 Sortie : {exit_type}")

    # Risk management
    risk = strategy.get("risk_management", {})
    if risk:
        lot = risk.get("lot_size", "?")
        max_risk = risk.get("max_risk_percent", "?")
        sizing = risk.get("position_sizing", "?")
        lines.append(f"      💰 Risque : {lot} lots, max {max_risk}% par trade ({sizing})")

    return lines


def _format_quick_test(result: Dict[str, Any]) -> str:
    """Formate le résultat d'un test rapide."""
    tested = result.get("tested", 0)
    viable = result.get("viable", 0)
    best_score = result.get("best_score", 0)
    results = result.get("results", [])

    lines = []
    lines.append("📊 **Test rapide de stratégies trading**")
    lines.append(f"{'═' * 55}")
    lines.append("")
    lines.append(f"  🎯 Stratégies testées : {tested}")
    lines.append(f"  ✅ Stratégies viables : {viable}/{tested}")
    lines.append(f"  🏆 Meilleur score     : {best_score:.1f}/100")
    lines.append("")

    if results:
        lines.append(f"{'─' * 55}")
        lines.append("📋 Détail des résultats :")

        for i, r in enumerate(results, 1):
            symbol = r.get("symbol", "?")
            tf = r.get("timeframe", "?")
            score = r.get("score", 0)
            profit = r.get("profit", 0)
            dd = r.get("drawdown", 0)
            trades = r.get("trades", 0)
            is_viable = r.get("viable", False)

            icon = "🟢" if is_viable else "🔴"
            profit_icon = "📈" if profit > 0 else "📉"

            lines.append("")
            lines.append(f"  {icon} #{i} — {symbol} {tf}")
            lines.append(f"      Score : {score:.1f}/100")
            lines.append(f"      {profit_icon} Profit : {profit:+.2f}$")
            lines.append(f"      📉 Drawdown : {dd:.1f}%")
            lines.append(f"      🔄 Trades : {trades}")

            # Détails de la stratégie si disponibles
            strategy = r.get("strategy", {})
            if strategy:
                detail_lines = _format_strategy_details(strategy)
                lines.extend(detail_lines)

            # Dates
            date_start = r.get("date_start", "")
            date_end = r.get("date_end", "")
            if date_start and date_end:
                lines.append(f"      📅 Période : {date_start} → {date_end}")

    lines.append("")
    lines.append(f"{'═' * 55}")
    if viable > 0:
        best = results[0] if results else {}
        lines.append(
            f"💡 Meilleure : {best.get('symbol', '?')} {best.get('timeframe', '?')} "
            f"→ {best.get('profit', 0):+.2f}$"
        )
        lines.append("   → Dis-moi : \"cherche une stratégie de trading\" pour optimiser")
    else:
        lines.append("⚠️ Aucune stratégie viable. Essaie une recherche optimisée !")
        lines.append("   → Dis-moi : \"cherche une stratégie de trading\"")

    return "\n".join(lines)


def _format_search(result: Dict[str, Any]) -> str:
    """Formate le résultat d'une recherche optimisée."""
    run_id = result.get("run_id", "?")
    best_score = result.get("best_score", 0)
    gens = result.get("generations_run", 0)
    total_tested = result.get("total_strategies_tested", 0)
    best_strat = result.get("best_strategy", {})
    best_metrics = result.get("best_metrics", {})
    history = result.get("convergence_history", [])

    lines = []
    lines.append("🔍 **Recherche de stratégie optimale**")
    lines.append(f"{'═' * 55}")
    lines.append("")
    lines.append(f"  🆔 Run              : {run_id}")
    lines.append(f"  🔄 Générations      : {gens}")
    lines.append(f"  📊 Stratégies testées: {total_tested}")
    lines.append(f"  🏆 Meilleur score   : {best_score:.1f}/100")
    lines.append("")

    if best_strat:
        lines.append(f"{'─' * 55}")
        lines.append("🥇 **Meilleure stratégie trouvée**")
        lines.append("")
        lines.append(f"  🆔 ID        : {best_strat.get('id', '?')}")
        lines.append(f"  💱 Symbole   : {best_strat.get('symbol', '?')}")
        lines.append(f"  ⏱️  Timeframe : {best_strat.get('timeframe', '?')}")
        lines.append("")

        # Détails complets de la stratégie
        detail_lines = _format_strategy_details(best_strat)
        # Remonter l'indentation d'un niveau
        for dl in detail_lines:
            lines.append(dl.replace("      ", "  "))
        lines.append("")

    if best_metrics:
        lines.append(f"{'─' * 55}")
        lines.append("📈 **Résultats du backtest**")
        lines.append("")

        profit = best_metrics.get("profit", 0)
        lines.append(f"  {'📈' if profit > 0 else '📉'} Profit        : {profit:+.2f}$")
        lines.append(f"  📉 Drawdown      : {best_metrics.get('drawdown', 0):.1f}%")
        lines.append(f"  🎯 Winrate       : {best_metrics.get('winrate', 0):.1f}%")
        lines.append(f"  🔄 Trades        : {best_metrics.get('trades', 0)}")
        lines.append(f"  📊 Profit Factor : {best_metrics.get('profit_factor', 0):.2f}")
        lines.append(f"  📏 Sharpe Ratio  : {best_metrics.get('sharpe_ratio', 0):.2f}")
        lines.append(f"  💰 Rendement     : {best_metrics.get('return_pct', 0):.1f}%")
        lines.append(f"  💵 Espérance/trade: {best_metrics.get('expectancy', 0):.2f}$")

        # Dates de la période
        date_start = best_metrics.get("date_start", "")
        date_end = best_metrics.get("date_end", "")
        if not date_start:
            # Chercher dans le backtest_result
            br = result.get("backtest_result", {})
            date_start = br.get("date_start", "")
            date_end = br.get("date_end", "")
        if date_start:
            lines.append(f"  📅 Période       : {date_start} → {date_end}")
        lines.append("")

    # Évolution
    if history and len(history) > 1:
        lines.append(f"{'─' * 55}")
        lines.append("📈 **Évolution du score**")
        lines.append("")
        points = _select_history_points(history, max_points=6)
        for h in points:
            bar_len = max(1, int(h["best_score"] / 5))
            bar = "█" * bar_len
            lines.append(
                f"  Gen {h['generation']:3d} │ {bar} {h['best_score']:.1f} "
                f"(avg: {h['avg_score']:.1f}, viables: {h['viable_count']})"
            )
        lines.append("")

    lines.append(f"{'═' * 55}")
    if best_score > 50:
        lines.append("✅ Excellente stratégie trouvée !")
        lines.append("   → \"améliore cette stratégie\" pour l'affiner")
    elif best_score > 25:
        lines.append("✅ Stratégie correcte, mais perfectible.")
        lines.append("   → \"cherche une stratégie\" pour relancer avec plus de générations")
    elif best_score > 0:
        lines.append("⚠️ Stratégie faible trouvée.")
        lines.append("   → Essaie avec plus de générations ou un autre symbole")
    else:
        lines.append("⚠️ Aucune bonne stratégie trouvée.")
        lines.append("   → Essaie avec un autre symbole ou timeframe")

    return "\n".join(lines)


def _format_evaluate(result: Dict[str, Any]) -> str:
    """Formate l'évaluation d'une stratégie."""
    strat_id = result.get("strategy_id", "?")
    score = result.get("score", 0)
    viable = result.get("viable", False)
    metrics = result.get("metrics", {})

    lines = []
    lines.append("📊 **Évaluation de stratégie**")
    lines.append(f"{'═' * 55}")
    lines.append("")
    lines.append(f"  🆔 ID     : {strat_id}")
    lines.append(f"  🏆 Score  : {score:.1f}/100")
    lines.append(f"  {'✅ Viable' if viable else '❌ Non viable'}")
    lines.append("")

    if metrics:
        profit = metrics.get("profit", 0)
        lines.append(f"  {'📈' if profit > 0 else '📉'} Profit    : {profit:+.2f}$")
        lines.append(f"  📉 Drawdown  : {metrics.get('drawdown', 0):.1f}%")
        lines.append(f"  🎯 Winrate   : {metrics.get('winrate', 0):.1f}%")
        lines.append(f"  🔄 Trades    : {metrics.get('trades', 0)}")

    return "\n".join(lines)


def _format_improve(result: Dict[str, Any]) -> str:
    """Formate le résultat d'amélioration."""
    original = result.get("original_score", 0)
    best = result.get("best_score", 0)
    improved = result.get("improved", False)
    improvement_pct = result.get("improvement_pct", 0)
    found = result.get("improvements_found", 0)
    iterations = result.get("iterations", 0)
    best_strat = result.get("best_strategy", {})

    lines = []
    lines.append("🔧 **Amélioration de stratégie**")
    lines.append(f"{'═' * 55}")
    lines.append("")
    lines.append(f"  🔄 Itérations       : {iterations}")
    lines.append(f"  📊 Score original   : {original:.1f}")
    lines.append(f"  🏆 Meilleur score   : {best:.1f}")
    lines.append(f"  {'📈' if improved else '➡️'} Améliorations    : {found}")

    if improved:
        lines.append(f"  ✅ Amélioration de {improvement_pct:+.1f}% !")
        lines.append("")

        if best_strat:
            lines.append(f"{'─' * 55}")
            lines.append("📋 **Stratégie améliorée**")
            lines.append("")
            detail_lines = _format_strategy_details(best_strat)
            for dl in detail_lines:
                lines.append(dl.replace("      ", "  "))
    else:
        lines.append("  ⚠️ Aucune amélioration trouvée")
        lines.append("  → Essaie avec plus d'itérations")

    return "\n".join(lines)


def _format_report(result: Dict[str, Any]) -> str:
    """Formate le rapport de session."""
    lines = []
    lines.append("📋 **Rapport de session trading**")
    lines.append(f"{'═' * 55}")
    lines.append("")
    lines.append(f"  🕐 Démarrage   : {result.get('session_start', '?')}")
    lines.append(f"  ⏱️  Durée       : {_format_duration(result.get('duration_seconds', 0))}")
    lines.append(f"  🔄 Recherches  : {result.get('total_optimization_runs', 0)}")
    lines.append(f"  🏆 Hall of Fame: {result.get('strategies_in_hall_of_fame', 0)} stratégies")
    lines.append(f"  ⭐ Meilleur    : {result.get('best_score_ever', 0):.1f}/100")
    lines.append("")

    top = result.get("top_5", [])
    if top:
        lines.append(f"{'─' * 55}")
        lines.append("🥇 **Top 5 stratégies**")
        lines.append("")
        for i, s in enumerate(top, 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(i-1, 4)]
            strat = s.get("strategy", {})
            metrics = s.get("metrics", {})
            lines.append(
                f"  {medal} Score {s.get('score', 0):.1f} — "
                f"{strat.get('symbol', '?')} {strat.get('timeframe', '?')} "
                f"(profit: {metrics.get('profit', 0):+.2f}$)"
            )

            detail_lines = _format_strategy_details(strat)
            lines.extend(detail_lines)
            lines.append("")
    else:
        lines.append("  📭 Aucune stratégie encore.")
        lines.append("  → \"lance un test rapide de trading\"")

    return "\n".join(lines)


def _format_top_strategies(result) -> str:
    """Formate la liste des meilleures stratégies."""
    if isinstance(result, dict) and "result" in result:
        strategies = result["result"]
    elif isinstance(result, list):
        strategies = result
    else:
        strategies = []

    lines = []
    lines.append("🏆 **Meilleures stratégies**")
    lines.append(f"{'═' * 55}")
    lines.append("")

    if not strategies:
        lines.append("  📭 Aucune stratégie dans le hall of fame.")
        lines.append("  → \"cherche une stratégie de trading\"")
        return "\n".join(lines)

    for i, s in enumerate(strategies, 1):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(i-1, 4)]
        strat = s.get("strategy", {})
        metrics = s.get("metrics", {})
        score = s.get("score", 0)
        profit = metrics.get("profit", 0)

        lines.append(f"  {medal} #{i} — {strat.get('symbol', '?')} {strat.get('timeframe', '?')}")
        lines.append(f"      Score : {score:.1f}/100")
        lines.append(f"      {'📈' if profit > 0 else '📉'} Profit : {profit:+.2f}$")
        lines.append(f"      Winrate : {metrics.get('winrate', 0):.1f}% | "
                      f"DD: {metrics.get('drawdown', 0):.1f}% | "
                      f"Trades: {metrics.get('trades', 0)}")

        detail_lines = _format_strategy_details(strat)
        lines.extend(detail_lines)
        lines.append("")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} secondes"
    elif seconds < 3600:
        return f"{int(seconds // 60)}min {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}min"


def _select_history_points(history: list, max_points: int = 6) -> list:
    if len(history) <= max_points:
        return history
    indices = [0]
    step = (len(history) - 1) / (max_points - 1)
    for i in range(1, max_points - 1):
        indices.append(int(i * step))
    indices.append(len(history) - 1)
    indices = sorted(set(indices))
    return [history[i] for i in indices]


def is_trading_tool(tool_name: str) -> bool:
    return tool_name in (
        "trading_quick_test", "trading_search", "trading_evaluate",
        "trading_improve", "trading_report", "trading_top_strategies",
    )