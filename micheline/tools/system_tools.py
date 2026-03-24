"""
System Tools — Outils système de base.
Calculs, date/heure, infos système, etc.
"""

import os
import sys
import math
import platform
from datetime import datetime


def tool_calculator(params: dict) -> dict:
    """Évalue une expression mathématique simple."""
    expression = params.get("expression", "")
    if not expression:
        return {"success": False, "error": "Pas d'expression fournie"}

    # Sécurité: n'autoriser que les caractères mathématiques
    allowed = set("0123456789+-*/().,%^ sincotaqrtlgexpabmdfoh ")
    if not all(c in allowed for c in expression.lower().replace("pi", "").replace("sqrt", "")):
        return {"success": False, "error": f"Expression non autorisée: {expression}"}

    try:
        # Remplacer les fonctions courantes
        safe_expr = expression.replace("^", "**")

        # Namespace sécurisé
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "floor": math.floor, "ceil": math.ceil,
        }

        result = eval(safe_expr, {"__builtins__": {}}, safe_dict)
        return {"success": True, "data": f"{expression} = {result}"}
    except Exception as e:
        return {"success": False, "error": f"Erreur calcul: {str(e)}"}


def tool_datetime(params: dict) -> dict:
    """Retourne la date et l'heure actuelles."""
    now = datetime.now()
    fmt = params.get("format", "%Y-%m-%d %H:%M:%S")
    try:
        formatted = now.strftime(fmt)
    except Exception:
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "success": True,
        "data": {
            "datetime": formatted,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": now.timestamp(),
            "weekday": now.strftime("%A"),
        }
    }


def tool_system_info(params: dict) -> dict:
    """Retourne des informations sur le système."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "cwd": os.getcwd(),
    }

    # RAM (si psutil dispo)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
        info["ram_used_percent"] = mem.percent
    except ImportError:
        info["ram"] = "(psutil non installé)"

    return {"success": True, "data": info}


def tool_list_directory(params: dict) -> dict:
    """Liste le contenu d'un répertoire (noms seulement, pas de contenu fichier)."""
    path = params.get("path", ".")

    try:
        if not os.path.isdir(path):
            return {"success": False, "error": f"Répertoire introuvable: {path}"}

        entries = []
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            entry_type = "dir" if os.path.isdir(full) else "file"
            size = 0
            if entry_type == "file":
                try:
                    size = os.path.getsize(full)
                except Exception:
                    pass
            entries.append({"name": entry, "type": entry_type, "size": size})

        return {
            "success": True,
            "data": {
                "path": os.path.abspath(path),
                "count": len(entries),
                "entries": entries[:100]  # Limite à 100 entrées
            }
        }
    except PermissionError:
        return {"success": False, "error": f"Accès refusé: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Erreur: {str(e)}"}


def tool_web_search_stub(params: dict) -> dict:
    """Stub pour recherche web (sera implémenté avec les vrais outils plus tard)."""
    query = params.get("query", "")
    return {
        "success": True,
        "data": f"Recherche web pour '{query}' - fonctionnalité en cours d'implémentation (Phase 5+)"
    }

def tool_memory_search(params: dict) -> dict:
    """Recherche dans la mémoire de l'agent."""
    try:
        from micheline.memory.memory import AgentMemory
        memory = AgentMemory()
        query = params.get("query", "")

        if not query:
            stats = memory.get_stats()
            return {"success": True, "data": stats}

        results = memory.search_experiences(query, limit=int(params.get("limit", 10)))
        return {
            "success": True,
            "data": {
                "query": query,
                "results_count": len(results),
                "results": results
            }
        }
    except Exception as e:
        return {"success": False, "error": f"Erreur mémoire: {str(e)}"}


def tool_memory_stats(params: dict) -> dict:
    """Retourne les statistiques de la mémoire de l'agent."""
    try:
        from micheline.memory.memory import AgentMemory
        memory = AgentMemory()
        return {"success": True, "data": memory.get_stats()}
    except Exception as e:
        return {"success": False, "error": f"Erreur: {str(e)}"}

def register_system_tools(registry):
    """Enregistre tous les outils système dans le registry."""

    registry.register(
        name="calculator",
        description="Évalue une expression mathématique (ex: 2+2, sqrt(16), sin(3.14))",
        function=tool_calculator,
        param_schema={"expression": "string - l'expression mathématique à évaluer"}
    )

    registry.register(
        name="datetime",
        description="Retourne la date et l'heure actuelles",
        function=tool_datetime,
        param_schema={"format": "string (optionnel) - format strftime"}
    )

    registry.register(
        name="system_info",
        description="Retourne les informations système (OS, RAM, Python)",
        function=tool_system_info,
        param_schema={}
    )

    registry.register(
        name="list_directory",
        description="Liste le contenu d'un répertoire (noms de fichiers/dossiers)",
        function=tool_list_directory,
        param_schema={"path": "string - chemin du répertoire à lister"}
    )

    registry.register(
        name="web_search",
        description="Recherche web (stub - sera implémenté plus tard)",
        function=tool_web_search_stub,
        param_schema={"query": "string - requête de recherche"}
    )

    registry.register(
        name="memory_search",
        description="Recherche dans la mémoire de l'agent (expériences passées)",
        function=tool_memory_search,
        param_schema={"query": "string - mots-clés à rechercher", "limit": "int (optionnel) - max résultats"}
    )

    registry.register(
        name="memory_stats",
        description="Retourne les statistiques de la mémoire (combien d'expériences, taux de succès)",
        function=tool_memory_stats,
        param_schema={}
    )

    print(f"[Tools] {len(registry.list_tools())} outils enregistrés.")