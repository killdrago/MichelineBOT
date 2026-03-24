"""
Tool Registry — Registre centralisé de tous les outils.
Emplacement : micheline/tools/registry.py
FICHIER MODIFIÉ — Phase 5
"""

import os
import json
import math
import platform
import psutil
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ToolRegistry:
    """Registre centralisé de tous les outils disponibles."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_tools()
        self._register_phase5_tools()
    
    def register(self, name: str, func: Callable, description: str, parameters: Dict[str, str]):
        """Enregistre un outil dans le registre."""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def execute(self, name: str, params: Dict[str, Any] = None) -> str:
        """Exécute un outil par son nom."""
        if name not in self.tools:
            available = ", ".join(sorted(self.tools.keys()))
            return f"Outil inconnu : '{name}'. Outils disponibles : {available}"
        
        if params is None:
            params = {}
        
        try:
            result = self.tools[name]["function"](**params)
            return str(result)
        except TypeError as e:
            return f"Erreur de paramètres pour '{name}': {e}. Paramètres attendus : {self.tools[name]['parameters']}"
        except Exception as e:
            return f"Erreur lors de l'exécution de '{name}': {type(e).__name__}: {e}"
    
    def list_tools(self) -> str:
        """Liste tous les outils disponibles."""
        if not self.tools:
            return "Aucun outil disponible."
        
        lines = ["🔧 OUTILS DISPONIBLES :", ""]
        for name, info in sorted(self.tools.items()):
            lines.append(f"  📌 {name}")
            lines.append(f"     {info['description']}")
            if info['parameters']:
                for param, desc in info['parameters'].items():
                    lines.append(f"       - {param}: {desc}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_tool_info(self, name: str) -> Optional[Dict]:
        """Retourne les infos d'un outil."""
        return self.tools.get(name)
    
    def get_tools_for_prompt(self) -> str:
        """Génère la description des outils pour le prompt du LLM."""
        lines = []
        for name, info in sorted(self.tools.items()):
            params_str = ", ".join(f"{k}: {v}" for k, v in info['parameters'].items())
            lines.append(f"- {name}({params_str}): {info['description']}")
        return "\n".join(lines)
    
    # =========================================================================
    # OUTILS INTÉGRÉS (Phases 1-4 — inchangés)
    # =========================================================================
    
    def _register_builtin_tools(self):
        """Enregistre les outils intégrés (ceux des phases 1-4)."""
        
        # --- Calculator ---
        def calculator(expression: str) -> str:
            safe_dict = {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sum": sum, "len": len,
                "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e, "inf": math.inf,
                "floor": math.floor, "ceil": math.ceil,
                "factorial": math.factorial, "gcd": math.gcd,
            }
            try:
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                return f"Résultat : {expression} = {result}"
            except Exception as e:
                return f"Erreur de calcul : {e}"
        
        self.register("calculator", calculator,
                       "Évalue une expression mathématique",
                       {"expression": "str — Expression mathématique (ex: 'sqrt(144)', '2**10')"})
        
        # --- DateTime ---
        def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
            now = dt.datetime.now()
            return f"Date/heure actuelle : {now.strftime(format)}"
        
        self.register("datetime", get_datetime,
                       "Retourne la date et l'heure actuelles",
                       {"format": "str — Format de date (optionnel, défaut: '%Y-%m-%d %H:%M:%S')"})
        
        # --- System Info ---
        def system_info() -> str:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return (
                    f"💻 Système : {platform.system()} {platform.release()}\n"
                    f"🖥️ Machine : {platform.machine()}\n"
                    f"📊 CPU : {cpu_percent}% utilisé ({psutil.cpu_count()} cœurs)\n"
                    f"🧠 RAM : {memory.percent}% utilisée "
                    f"({memory.used // (1024**3)}/{memory.total // (1024**3)} GB)\n"
                    f"💾 Disque : {disk.percent}% utilisé "
                    f"({disk.used // (1024**3)}/{disk.total // (1024**3)} GB)"
                )
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("system_info", system_info,
                       "Affiche les informations système (CPU, RAM, disque)",
                       {})
        
        # --- List Directory ---
        def list_directory(path: str = ".") -> str:
            try:
                from micheline.security.path_guard import is_allowed
                if not is_allowed(path):
                    return f"🚫 Accès refusé : {path}"
            except ImportError:
                pass
            
            try:
                p = Path(path)
                if not p.exists():
                    return f"Chemin inexistant : {path}"
                if not p.is_dir():
                    return f"N'est pas un dossier : {path}"
                
                items = sorted(p.iterdir())
                dirs = [f"📁 {item.name}/" for item in items if item.is_dir()]
                files = [f"📄 {item.name}" for item in items if item.is_file()]
                
                result = f"📂 Contenu de {path} ({len(dirs)} dossiers, {len(files)} fichiers) :\n"
                result += "\n".join(dirs + files)
                return result
            except PermissionError:
                return f"🚫 Permission refusée : {path}"
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("list_directory", list_directory,
                       "Liste le contenu d'un dossier",
                       {"path": "str — Chemin du dossier"})
        
        # --- Read File ---
        def read_file(path: str, max_lines: int = 100) -> str:
            try:
                from micheline.security.path_guard import is_allowed
                if not is_allowed(path):
                    return f"🚫 Accès refusé : {path}"
            except ImportError:
                pass
            
            try:
                p = Path(path)
                if not p.exists():
                    return f"Fichier inexistant : {path}"
                if not p.is_file():
                    return f"N'est pas un fichier : {path}"
                
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                
                total = len(lines)
                content = "".join(lines[:max_lines])
                
                result = f"📄 {path} ({total} lignes) :\n{content}"
                if total > max_lines:
                    result += f"\n... [{total - max_lines} lignes supplémentaires]"
                return result
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("read_file", read_file,
                       "Lit le contenu d'un fichier",
                       {"path": "str — Chemin du fichier", "max_lines": "int — Nombre max de lignes (défaut: 100)"})
        
        # --- Write File ---
        def write_file(path: str, content: str) -> str:
            try:
                from micheline.security.path_guard import is_allowed
                if not is_allowed(path):
                    return f"🚫 Accès refusé : {path}"
            except ImportError:
                pass
            
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"✅ Fichier écrit : {path} ({len(content)} caractères)"
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("write_file", write_file,
                       "Écrit du contenu dans un fichier",
                       {"path": "str — Chemin du fichier", "content": "str — Contenu à écrire"})
        
        # --- File Info ---
        def file_info(path: str) -> str:
            try:
                from micheline.security.path_guard import is_allowed
                if not is_allowed(path):
                    return f"🚫 Accès refusé : {path}"
            except ImportError:
                pass
            
            try:
                p = Path(path)
                if not p.exists():
                    return f"Inexistant : {path}"
                
                stat = p.stat()
                size = stat.st_size
                modified = dt.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                if size < 1024:
                    size_str = f"{size} octets"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024*1024):.1f} MB"
                
                return (
                    f"📄 {path}\n"
                    f"  Type : {'dossier' if p.is_dir() else 'fichier'}\n"
                    f"  Taille : {size_str}\n"
                    f"  Modifié : {modified}\n"
                    f"  Extension : {p.suffix or 'aucune'}"
                )
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("file_info", file_info,
                       "Affiche les informations d'un fichier",
                       {"path": "str — Chemin du fichier"})
        
        # --- Memory Search ---
        def memory_search(query: str = "") -> str:
            try:
                from micheline.memory.memory import memory_manager
                if not query:
                    stats = memory_manager.get_stats()
                    return f"📊 Mémoire : {json.dumps(stats, indent=2, ensure_ascii=False)}"
                results = memory_manager.search(query)
                if not results:
                    return f"Aucun souvenir trouvé pour '{query}'."
                formatted = [f"🧠 Résultats pour '{query}' :"]
                for r in results[:5]:
                    formatted.append(f"  - [{r.get('type', '?')}] {r.get('content', '')[:100]}")
                return "\n".join(formatted)
            except Exception as e:
                return f"Erreur mémoire : {e}"
        
        self.register("memory_search", memory_search,
                       "Recherche dans la mémoire persistante",
                       {"query": "str — Terme de recherche (vide = stats)"})
        
        # --- Memory Stats ---
        def memory_stats() -> str:
            try:
                from micheline.memory.memory import memory_manager
                stats = memory_manager.get_stats()
                return f"📊 Statistiques mémoire :\n{json.dumps(stats, indent=2, ensure_ascii=False)}"
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("memory_stats", memory_stats,
                       "Affiche les statistiques de la mémoire",
                       {})
        
        # --- List Allowed Paths ---
        def list_allowed_paths() -> str:
            try:
                from micheline.security.path_guard import get_allowed_paths_display
                paths = get_allowed_paths_display()
                result = "📁 Chemins autorisés :\n"
                for p in paths:
                    result += f"  ✅ {p}\n"
                return result
            except ImportError:
                return "Module de sécurité non disponible."
            except Exception as e:
                return f"Erreur : {e}"
        
        self.register("list_allowed_paths", list_allowed_paths,
                       "Liste les chemins autorisés par le système de sécurité",
                       {})
    
    # =========================================================================
    # OUTILS PHASE 5 — Nouveaux
    # =========================================================================
    
    def _register_phase5_tools(self):
        """Enregistre les outils de la Phase 5."""
        
        # --- Code Executor ---
        try:
            from micheline.tools.code_executor import execute_code, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, execute_code, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] Code Executor non chargé : {e}")
        
        # --- Web Search ---
        try:
            from micheline.tools.web_search_tool import web_search, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, web_search, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] Web Search non chargé : {e}")
        
        # --- Shell Command ---
        try:
            from micheline.tools.shell_tool import shell_command, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, shell_command, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] Shell Tool non chargé : {e}")
        
        # --- MT5 Tool ---
        try:
            from micheline.tools.mt5_tool import mt5_tool, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, mt5_tool, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] MT5 Tool non chargé : {e}")
        
        # --- Task Planner ---
        try:
            from micheline.tools.task_planner_tool import task_planner, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, task_planner, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] Task Planner non chargé : {e}")
            
        # --- App Launcher ---
        try:
            from micheline.tools.app_launcher import app_launcher, TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS
            self.register(TOOL_NAME, app_launcher, TOOL_DESCRIPTION, TOOL_PARAMETERS)
        except ImportError as e:
            print(f"[Registry] App Launcher non chargé : {e}")


# Instance globale du registre
tool_registry = ToolRegistry()