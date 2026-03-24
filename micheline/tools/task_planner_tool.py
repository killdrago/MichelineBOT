"""
Task Planner Tool — Décompose un problème complexe en sous-tâches.
Emplacement : micheline/tools/task_planner_tool.py
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class TaskDecomposer:
    """Décompose des problèmes complexes en étapes actionnables."""
    
    # Templates de décomposition par domaine
    TEMPLATES = {
        "trading_strategy": {
            "description": "Recherche et optimisation de stratégie trading",
            "steps": [
                {"id": 1, "action": "mt5_tool", "params": {"action": "connect"}, "description": "Connecter à MT5"},
                {"id": 2, "action": "mt5_tool", "params": {"action": "account_info"}, "description": "Vérifier le compte"},
                {"id": 3, "action": "mt5_tool", "params": {"action": "historical_data"}, "description": "Récupérer données historiques"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Analyser les données et calculer indicateurs"},
                {"id": 5, "action": "code_executor", "params": {}, "description": "Backtester la stratégie"},
                {"id": 6, "action": "memory_search", "params": {}, "description": "Comparer avec stratégies précédentes"},
                {"id": 7, "action": "code_executor", "params": {}, "description": "Optimiser les paramètres"},
                {"id": 8, "action": "write_file", "params": {}, "description": "Sauvegarder les résultats"},
            ]
        },
        "data_analysis": {
            "description": "Analyse de données",
            "steps": [
                {"id": 1, "action": "read_file", "params": {}, "description": "Lire le fichier de données"},
                {"id": 2, "action": "code_executor", "params": {}, "description": "Explorer et nettoyer les données"},
                {"id": 3, "action": "code_executor", "params": {}, "description": "Calculer les statistiques"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Visualiser les résultats"},
                {"id": 5, "action": "write_file", "params": {}, "description": "Sauvegarder le rapport"},
            ]
        },
        "research": {
            "description": "Recherche d'information",
            "steps": [
                {"id": 1, "action": "web_search", "params": {"source": "all"}, "description": "Rechercher des sources"},
                {"id": 2, "action": "web_search", "params": {"source": "wiki_summary"}, "description": "Approfondir les résultats clés"},
                {"id": 3, "action": "memory_search", "params": {}, "description": "Vérifier connaissances existantes"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Synthétiser l'information"},
                {"id": 5, "action": "write_file", "params": {}, "description": "Rédiger le rapport"},
            ]
        },
        "programming": {
            "description": "Résolution de problème de programmation",
            "steps": [
                {"id": 1, "action": "code_executor", "params": {}, "description": "Analyser le problème"},
                {"id": 2, "action": "code_executor", "params": {}, "description": "Implémenter une première solution"},
                {"id": 3, "action": "code_executor", "params": {}, "description": "Tester avec des cas limites"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Optimiser si nécessaire"},
                {"id": 5, "action": "write_file", "params": {}, "description": "Sauvegarder le code final"},
            ]
        },
        "system_diagnostic": {
            "description": "Diagnostic système",
            "steps": [
                {"id": 1, "action": "system_info", "params": {}, "description": "Collecter infos système"},
                {"id": 2, "action": "shell_command", "params": {}, "description": "Vérifier processus et services"},
                {"id": 3, "action": "shell_command", "params": {}, "description": "Vérifier espace disque et réseau"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Analyser les résultats"},
                {"id": 5, "action": "write_file", "params": {}, "description": "Générer rapport diagnostic"},
            ]
        },
        "generic": {
            "description": "Résolution de problème générique",
            "steps": [
                {"id": 1, "action": "memory_search", "params": {}, "description": "Vérifier si un problème similaire a été résolu"},
                {"id": 2, "action": "web_search", "params": {}, "description": "Rechercher des informations pertinentes"},
                {"id": 3, "action": "code_executor", "params": {}, "description": "Analyser et calculer"},
                {"id": 4, "action": "code_executor", "params": {}, "description": "Implémenter la solution"},
                {"id": 5, "action": "write_file", "params": {}, "description": "Documenter le résultat"},
            ]
        }
    }
    
    # Mots-clés pour détecter le domaine
    DOMAIN_KEYWORDS = {
        "trading_strategy": [
            "trading", "stratégie", "strategy", "backtest", "mt5", "metatrader",
            "forex", "bourse", "action", "crypto", "bitcoin", "eurusd",
            "indicateur", "rsi", "macd", "moving average", "bollinger"
        ],
        "data_analysis": [
            "analyse", "données", "data", "csv", "excel", "statistique",
            "graphique", "visualisation", "corrélation", "régression",
            "moyenne", "médiane", "distribution"
        ],
        "research": [
            "recherche", "trouve", "information", "news", "actualité",
            "wikipedia", "article", "sujet", "explique", "c'est quoi",
            "qu'est-ce que", "histoire de"
        ],
        "programming": [
            "code", "programme", "script", "algorithme", "fonction",
            "python", "trie", "boucle", "récursif", "classe",
            "bug", "erreur", "debug", "optimise le code"
        ],
        "system_diagnostic": [
            "système", "diagnostic", "performance", "mémoire", "cpu",
            "disque", "processus", "service", "réseau", "ping"
        ]
    }
    
    def detect_domain(self, problem: str) -> str:
        """Détecte le domaine d'un problème."""
        problem_lower = problem.lower()
        
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in problem_lower)
            scores[domain] = score
        
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] == 0:
            return "generic"
        
        return best_domain
    
    def decompose(self, problem: str, domain: str = None) -> Dict[str, Any]:
        """
        Décompose un problème en sous-tâches.
        
        Args:
            problem: Description du problème
            domain: Domaine forcé (optionnel, sinon auto-détecté)
        
        Returns:
            Plan structuré avec les étapes
        """
        if not domain:
            domain = self.detect_domain(problem)
        
        if domain not in self.TEMPLATES:
            domain = "generic"
        
        template = self.TEMPLATES[domain]
        
        return {
            "problem": problem,
            "domain": domain,
            "domain_description": template["description"],
            "total_steps": len(template["steps"]),
            "steps": template["steps"],
            "estimated_tools": list(set(step["action"] for step in template["steps"])),
            "created_at": datetime.now().isoformat()
        }


# Instance globale
_decomposer = TaskDecomposer()


def task_planner(problem: str, domain: str = None) -> str:
    """
    Point d'entrée pour le tool registry.
    
    Args:
        problem: Description du problème à résoudre
        domain: Domaine forcé (optionnel) — 'trading_strategy', 'data_analysis', 
                'research', 'programming', 'system_diagnostic', 'generic'
    
    Returns:
        Plan formaté en texte
    """
    if not problem or not problem.strip():
        return "Erreur : aucun problème décrit."
    
    plan = _decomposer.decompose(problem.strip(), domain)
    
    parts = [
        f"📋 PLAN DE RÉSOLUTION",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"🎯 Problème : {plan['problem']}",
        f"🏷️ Domaine détecté : {plan['domain']} ({plan['domain_description']})",
        f"📊 Étapes : {plan['total_steps']}",
        f"🔧 Outils nécessaires : {', '.join(plan['estimated_tools'])}",
        f"",
        f"📝 ÉTAPES :",
    ]
    
    for step in plan['steps']:
        parts.append(f"  {step['id']}. [{step['action']}] {step['description']}")
    
    parts.append(f"\n⏰ Plan créé le : {plan['created_at'][:19]}")
    
    return "\n".join(parts)


# Métadonnées pour le registry
TOOL_NAME = "task_planner"
TOOL_DESCRIPTION = (
    "Décompose un problème complexe en sous-tâches actionnables. "
    "Détecte automatiquement le domaine (trading, analyse de données, "
    "recherche, programmation, diagnostic système) et propose un plan "
    "étape par étape avec les outils nécessaires."
)
TOOL_PARAMETERS = {
    "problem": "str — Description du problème à résoudre",
    "domain": "str — Domaine forcé (optionnel) : 'trading_strategy', 'data_analysis', 'research', 'programming', 'system_diagnostic', 'generic'"
}