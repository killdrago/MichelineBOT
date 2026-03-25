# À AJOUTER dans core/agent_loop.py
# ──────────────────────────────────────────────
# Nouvelles capacités trading pour l'agent
# ──────────────────────────────────────────────

def handle_trading_objective(objective: str) -> dict:
    """
    Traite un objectif lié au trading.
    Appelé par la boucle principale quand l'objectif concerne le trading.

    Args:
        objective: description de l'objectif

    Returns:
        résultat de l'action trading
    """
    from tools.registry import execute_tool

    objective_lower = objective.lower()

    # Recherche de stratégie
    if any(word in objective_lower for word in ["cherche", "search", "trouve", "find", "optimise"]):
        params = _extract_trading_params(objective)
        result = execute_tool("trading_search", params)
        return {
            "action": "trading_search",
            "result": result,
            "summary": _summarize_trading_result(result),
        }

    # Test rapide
    elif any(word in objective_lower for word in ["test", "quick", "rapide", "essai"]):
        count = _extract_count(objective, default=5)
        result = execute_tool("trading_quick_test", {"count": count})
        return {
            "action": "trading_quick_test",
            "result": result,
            "summary": f"Test rapide : {result.get('viable', 0)}/{result.get('tested', 0)} viables",
        }

    # Rapport
    elif any(word in objective_lower for word in ["rapport", "report", "résumé", "summary", "status"]):
        result = execute_tool("trading_report")
        return {
            "action": "trading_report",
            "result": result,
        }

    # Top stratégies
    elif any(word in objective_lower for word in ["top", "meilleur", "best", "classement"]):
        count = _extract_count(objective, default=5)
        result = execute_tool("trading_top_strategies", {"count": count})
        return {
            "action": "trading_top_strategies",
            "result": result,
        }

    # Par défaut : recherche standard
    else:
        result = execute_tool("trading_search")
        return {
            "action": "trading_search",
            "result": result,
            "summary": _summarize_trading_result(result),
        }


def _extract_trading_params(objective: str) -> dict:
    """Extrait les paramètres de trading depuis l'objectif textuel."""
    import re

    params = {}

    # Chercher un symbole
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US500"]
    for symbol in symbols:
        if symbol.lower() in objective.lower():
            params["symbols"] = [symbol]
            break

    # Chercher un timeframe
    timeframes = {"m1": "M1", "m5": "M5", "m15": "M15", "m30": "M30",
                  "h1": "H1", "h4": "H4", "d1": "D1"}
    for key, value in timeframes.items():
        if key in objective.lower():
            params["timeframes"] = [value]
            break

    # Chercher un nombre de générations
    gen_match = re.search(r'(\d+)\s*(?:gen|génération|generation|iter)', objective.lower())
    if gen_match:
        params["max_generations"] = int(gen_match.group(1))

    # Chercher population size
    pop_match = re.search(r'(\d+)\s*(?:pop|population|strat)', objective.lower())
    if pop_match:
        params["population_size"] = int(pop_match.group(1))

    return params


def _extract_count(objective: str, default: int = 5) -> int:
    """Extrait un nombre depuis l'objectif."""
    import re
    match = re.search(r'(\d+)', objective)
    return int(match.group(1)) if match else default


def _summarize_trading_result(result: dict) -> str:
    """Crée un résumé lisible du résultat trading."""
    if "error" in result:
        return f"Erreur : {result['error']}"

    best_score = result.get("best_score", 0)
    tested = result.get("total_strategies_tested", 0)
    gens = result.get("generations_run", 0)

    best = result.get("best_strategy", {})
    best_id = best.get("id", "?") if best else "?"
    symbol = best.get("symbol", "?") if best else "?"

    metrics = result.get("best_metrics", {})
    profit = metrics.get("profit", 0)
    winrate = metrics.get("winrate", 0)

    return (
        f"Recherche terminée : {tested} stratégies testées en {gens} générations. "
        f"Meilleure : {best_id} ({symbol}) — "
        f"Score: {best_score:.1f}, Profit: {profit:.2f}$, Winrate: {winrate:.1f}%"
    )


def is_trading_objective(objective: str) -> bool:
    """Détermine si l'objectif concerne le trading."""
    trading_keywords = [
        "trading", "trade", "stratégie", "strategy", "backtest",
        "forex", "mt5", "metatrader", "optimis", "profit",
        "eurusd", "gbpusd", "xauusd", "bourse", "marché",
    ]
    return any(kw in objective.lower() for kw in trading_keywords)