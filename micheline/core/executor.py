"""
Executor — Exécute les actions planifiées.
Emplacement : micheline/core/executor.py
FICHIER MODIFIÉ — Phase 5
"""

import traceback
from typing import Dict, Any
from micheline.tools.registry import tool_registry


class Executor:
    """Exécute les actions définies par le Planner."""
    
    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute un plan d'action.
        
        Args:
            plan: Dictionnaire avec 'tool', 'params', et optionnellement 'reasoning'
        
        Returns:
            {
                "success": bool,
                "tool_used": str,
                "result": str,
                "error": str | None
            }
        """
        tool_name = plan.get("tool", "conversation")
        params = plan.get("params", {})
        
        # Cas spécial : conversation (pas d'outil à exécuter)
        if tool_name == "conversation":
            return {
                "success": True,
                "tool_used": "conversation",
                "result": params.get("response", "Je n'ai pas de réponse."),
                "error": None
            }
        
        # Vérifier que l'outil existe
        tool_info = tool_registry.get_tool_info(tool_name)
        if not tool_info:
            # Essayer avec des variantes de nom
            name_variants = {
                "search": "web_search",
                "web": "web_search",
                "execute_code": "code_executor",
                "run_code": "code_executor",
                "python": "code_executor",
                "shell": "shell_command",
                "cmd": "shell_command",
                "terminal": "shell_command",
                "mt5": "mt5_tool",
                "metatrader": "mt5_tool",
                "trading": "mt5_tool",
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
            }
            
            resolved_name = name_variants.get(tool_name.lower())
            if resolved_name and tool_registry.get_tool_info(resolved_name):
                tool_name = resolved_name
            else:
                return {
                    "success": False,
                    "tool_used": tool_name,
                    "result": None,
                    "error": f"Outil inconnu : '{tool_name}'. Disponibles : {', '.join(sorted(tool_registry.tools.keys()))}"
                }
        
        # Exécuter l'outil
        try:
            result = tool_registry.execute(tool_name, params)
            
            return {
                "success": True,
                "tool_used": tool_name,
                "result": result,
                "error": None
            }
        
        except Exception as e:
            error_detail = traceback.format_exc()
            return {
                "success": False,
                "tool_used": tool_name,
                "result": None,
                "error": f"{type(e).__name__}: {e}\n{error_detail}"
            }


# Instance globale
executor = Executor()