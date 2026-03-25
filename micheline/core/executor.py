"""
Executor — Exécute les actions planifiées.
Emplacement : micheline/core/executor.py
FICHIER MODIFIÉ — Phase 5 + Phase 6 (Trading Engine)
"""

import traceback
from typing import Dict, Any
from micheline.tools.registry import tool_registry


class Executor:
    """Exécute les actions définies par le Planner."""

    # Aliases de noms d'outils pour résolution
    NAME_VARIANTS = {
        # Phase 1-4
        "search": "web_search",
        "web": "web_search",
        "execute_code": "code_executor",
        "run_code": "code_executor",
        "python": "code_executor",
        "shell": "shell_command",
        "cmd": "shell_command",
        "terminal": "shell_command",
        "plan": "task_planner",
        "planner": "task_planner",
        "decompose": "task_planner",
        "calc": "calculator",
        "math": "calculator",
        "time": "datetime",
        "date": "datetime",
        "memory": "memory_search",
        "remember": "memory_search",
        "ls": "list_directory",
        "dir": "list_directory",
        "cat": "read_file",
        "read": "read_file",
        "write": "write_file",
        "save": "write_file",
        "sysinfo": "system_info",
        # Phase 5 — MT5
        "mt5": "initialize_mt5",
        "metatrader": "initialize_mt5",
        "mt5_tool": "initialize_mt5",
        # Phase 6 — Trading (aliases courts)
        "trading": "trading_quick_test",
        "trading_test": "trading_quick_test",
        "quick_test": "trading_quick_test",
        "trade_search": "trading_search",
        "trade_find": "trading_search",
        "trade_improve": "trading_improve",
        "trade_report": "trading_report",
        "trade_top": "trading_top_strategies",
        "trade_evaluate": "trading_evaluate",
    }

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute un plan d'action.

        Args:
            plan: Dictionnaire avec 'tool', 'params', et optionnellement 'reasoning'

        Returns:
            {
                "success": bool,
                "tool_used": str,
                "result": str | dict,
                "error": str | None
            }
        """
        tool_name = plan.get("tool", "conversation")
        params = plan.get("params", {})

        # Cas spécial : conversation
        if tool_name == "conversation":
            return {
                "success": True,
                "tool_used": "conversation",
                "result": params.get("response", "Je n'ai pas de réponse."),
                "error": None
            }

        # ── Résolution du nom de l'outil ──
        resolved_name = self._resolve_tool_name(tool_name)

        if resolved_name is None:
            available = ', '.join(sorted(tool_registry.tools.keys()))
            return {
                "success": False,
                "tool_used": tool_name,
                "result": None,
                "error": f"Outil inconnu : '{tool_name}'. Disponibles : {available}"
            }

        # ── Exécution ──
        try:
            result = tool_registry.execute(resolved_name, params)

            # Vérifier si le résultat contient une erreur du registry
            if isinstance(result, dict) and "error" in result and len(result) <= 2:
                return {
                    "success": False,
                    "tool_used": resolved_name,
                    "result": result.get("error", "Erreur inconnue"),
                    "error": result.get("error")
                }

            return {
                "success": True,
                "tool_used": resolved_name,
                "result": result,
                "error": None
            }

        except Exception as e:
            error_detail = traceback.format_exc()
            return {
                "success": False,
                "tool_used": resolved_name,
                "result": None,
                "error": f"{type(e).__name__}: {e}\n{error_detail}"
            }

    def _resolve_tool_name(self, tool_name: str) -> str:
        """
        Résout le nom d'un outil.
        1. Vérifie si le nom exact existe dans le registry
        2. Sinon, cherche dans les aliases
        3. Sinon, retourne None
        """
        # 1. Nom exact
        if tool_registry.has_tool(tool_name):
            return tool_name

        # 2. Aliases
        alias = self.NAME_VARIANTS.get(tool_name.lower())
        if alias and tool_registry.has_tool(alias):
            return alias

        # 3. Recherche partielle (pour les cas comme "mt5_tool" → tools qui commencent par "mt5")
        tool_lower = tool_name.lower()
        for registered_name in tool_registry.tool_names:
            if registered_name.lower() == tool_lower:
                return registered_name

        return None


# Instance globale
executor = Executor()