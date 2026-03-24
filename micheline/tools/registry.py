"""
Tool Registry — Système centralisé d'exécution d'outils.
Tous les accès au système passent par ce registre.
Chaque outil est une fonction enregistrée avec nom + description.
"""

from typing import Callable, Dict, Any, Optional
from datetime import datetime


class ToolResult:
    """Résultat standardisé d'un outil."""

    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }

    def __str__(self):
        if self.success:
            return str(self.data)[:500] if self.data else "(OK, pas de données)"
        return f"ERREUR: {self.error}"


class ToolDefinition:
    """Définition d'un outil enregistré."""

    def __init__(self, name: str, description: str, function: Callable,
                 param_schema: dict = None, requires_security: bool = False):
        self.name = name
        self.description = description
        self.function = function
        self.param_schema = param_schema or {}
        self.requires_security = requires_security
        self.call_count = 0
        self.last_called = None

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "params": self.param_schema,
            "requires_security": self.requires_security,
            "call_count": self.call_count
        }


class ToolRegistry:
    """
    Registre central de tous les outils disponibles.

    Usage:
        registry = ToolRegistry()
        registry.register("calculator", "Fait des calculs", my_calc_func)
        result = registry.execute("calculator", {"expression": "2+2"})
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._execution_log: list = []
        self._max_log_size = 1000

    def register(self, name: str, description: str, function: Callable,
                 param_schema: dict = None, requires_security: bool = False):
        """Enregistre un nouvel outil."""
        if name in self._tools:
            print(f"[ToolRegistry] ⚠️ Outil '{name}' déjà enregistré, écrasement.")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            function=function,
            param_schema=param_schema,
            requires_security=requires_security
        )
        print(f"[ToolRegistry] ✅ Outil enregistré: {name}")

    def unregister(self, name: str):
        """Supprime un outil du registre."""
        if name in self._tools:
            del self._tools[name]
            print(f"[ToolRegistry] Outil supprimé: {name}")

    def execute(self, name: str, params: dict = None) -> ToolResult:
        """
        Exécute un outil par son nom avec les paramètres donnés.
        Point d'entrée UNIQUE pour toute exécution d'outil.
        """
        params = params or {}

        # Vérifier que l'outil existe
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "(aucun)"
            error = f"Outil inconnu: '{name}'. Disponibles: {available}"
            self._log_execution(name, params, False, error)
            return ToolResult(success=False, error=error)

        tool = self._tools[name]

        # Exécuter
        try:
            result = tool.function(params)

            # Normaliser le résultat
            if isinstance(result, ToolResult):
                tool_result = result
            elif isinstance(result, dict):
                tool_result = ToolResult(
                    success=result.get("success", True),
                    data=result.get("data", result),
                    error=result.get("error")
                )
            elif isinstance(result, str):
                tool_result = ToolResult(success=True, data=result)
            else:
                tool_result = ToolResult(success=True, data=result)

            # Stats
            tool.call_count += 1
            tool.last_called = datetime.now().isoformat()

            self._log_execution(name, params, tool_result.success, str(tool_result))
            return tool_result

        except PermissionError as e:
            error = f"🔒 Accès refusé: {str(e)}"
            self._log_execution(name, params, False, error)
            return ToolResult(success=False, error=error)

        except Exception as e:
            error = f"Erreur exécution '{name}': {type(e).__name__}: {str(e)}"
            self._log_execution(name, params, False, error)
            return ToolResult(success=False, error=error)

    def list_tools(self) -> list:
        """Retourne la liste des noms d'outils disponibles."""
        return list(self._tools.keys())

    def list_tools_detailed(self) -> list:
        """Retourne les détails de tous les outils pour le LLM."""
        return [tool.to_dict() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """Retourne une description textuelle de tous les outils (pour le prompt LLM)."""
        if not self._tools:
            return "Aucun outil disponible."

        lines = []
        for name, tool in self._tools.items():
            params_desc = ""
            if tool.param_schema:
                params_desc = " | Params: " + ", ".join(
                    f"{k}: {v}" for k, v in tool.param_schema.items()
                )
            security = " 🔒" if tool.requires_security else ""
            lines.append(f"  - {name}: {tool.description}{params_desc}{security}")

        return "Outils disponibles:\n" + "\n".join(lines)

    def _log_execution(self, name: str, params: dict, success: bool, detail: str):
        """Log interne des exécutions."""
        entry = {
            "tool": name,
            "params": params,
            "success": success,
            "detail": detail[:300],
            "timestamp": datetime.now().isoformat()
        }
        self._execution_log.append(entry)

        # Rotation
        if len(self._execution_log) > self._max_log_size:
            self._execution_log = self._execution_log[-500:]

    def get_execution_log(self, last_n: int = 20) -> list:
        """Retourne les N dernières exécutions."""
        return self._execution_log[-last_n:]