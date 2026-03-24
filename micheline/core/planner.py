"""
Planner — Planifie les actions de l'agent.
Emplacement : micheline/core/planner.py
FICHIER MODIFIÉ — Phase 5
"""

import json
import re
from typing import Dict, Any, Optional, List


class Planner:
    """Planifie les actions en fonction de l'objectif et de la réponse du LLM."""
    
    # Patterns de détection d'outils par mots-clés
    TOOL_PATTERNS = {
        # Phase 1-4 outils
        "calculator": [
            r"calcul", r"combien\s+fait", r"résultat\s+de", r"somme", r"addition",
            r"soustraction", r"multiplication", r"division", r"racine",
            r"sqrt", r"pourcentage", r"\d+\s*[\+\-\*/\^]\s*\d+",
            r"puissance", r"factorielle", r"logarithme"
        ],
        "datetime": [
            r"heure", r"date", r"jour", r"quel\s+jour", r"quelle\s+heure",
            r"maintenant", r"aujourd'hui", r"timestamp"
        ],
        "system_info": [
            r"syst[eè]me", r"cpu", r"ram", r"m[eé]moire\s+vive", r"disque",
            r"processeur", r"info\s+syst", r"performance"
        ],
        "list_directory": [
            r"liste.*dossier", r"contenu.*dossier", r"qu.*(est|y\s*a).*dans.*dossier",
            r"fichiers\s+dans", r"ls\b", r"dir\b", r"explorer\s+le\s+dossier"
        ],
        "read_file": [
            r"li[st].*fichier", r"contenu.*fichier", r"ouvr.*fichier",
            r"affiche.*fichier", r"montre.*fichier", r"cat\b"
        ],
        "write_file": [
            r"[eé]cri[st].*fichier", r"sauvegard", r"cr[eé]e.*fichier",
            r"enregistr", r"note.*dans"
        ],
        "file_info": [
            r"info.*fichier", r"taille.*fichier", r"quand.*modifi",
            r"d[eé]tail.*fichier"
        ],
        "memory_search": [
            r"m[eé]moire", r"souvenir", r"rappel", r"tu\s+te\s+souviens",
            r"qu.*tu\s+sais", r"exp[eé]rience", r"d[eé]couvert"
        ],
        "memory_stats": [
            r"stat.*m[eé]moire", r"combien.*m[eé]moire", r"état.*m[eé]moire"
        ],
        "list_allowed_paths": [
            r"chemin.*autoris", r"dossier.*autoris", r"o[uù].*acc[eè]s",
            r"permission.*fichier", r"path.*allowed"
        ],
        # Phase 5 outils
        "code_executor": [
            r"ex[eé]cut.*code", r"lance.*python", r"programme.*python",
            r"script.*python", r"run\s+code", r"teste.*code",
            r"ex[eé]cute.*script", r"code\s+python",
            r"impl[eé]ment", r"algorithme", r"r[eé]sou[ds].*probl[eè]me.*programm",
            r"tri.*liste", r"fibonacci", r"factori",
            r"boucle", r"r[eé]cursif", r"classe.*python"
        ],
        "web_search": [
            r"recherch.*web", r"cherch.*internet", r"news", r"actualit",
            r"derni[eè]re.*nouvelle", r"wikipedia", r"article.*sur",
            r"info.*sur\s+l", r"c.est\s+quoi", r"qu.est.ce\s+que",
            r"trouve.*info", r"recherch.*sur"
        ],
        "shell_command": [
            r"command.*syst[eè]me", r"terminal", r"cmd", r"shell",
            r"lance.*command", r"ex[eé]cut.*command",
            r"ping\s+", r"ipconfig", r"tasklist", r"systeminfo",
            r"version.*python", r"pip\s+list", r"git\s+"
        ],
        "mt5_tool": [
            r"mt5", r"metatrader", r"trading", r"forex", r"bourse",
            r"connect.*mt5", r"position.*trading", r"compte.*trading",
            r"symbole.*trading", r"eurusd", r"btcusd", r"donn[eé]es.*march",
            r"prix.*actuel", r"cours\s+de", r"backtest",
            r"strat[eé]gie.*trading"
        ],
        "task_planner": [
            r"d[eé]compos.*probl[eè]me", r"plan.*action", r"[eé]tape.*par.*[eé]tape",
            r"comment.*proc[eé]der", r"planifi", r"fais.*plan",
            r"organis.*t[aâ]che", r"projet.*complexe"
        ],
    }
    
    def create_plan(self, objective: str, llm_response: str) -> Dict[str, Any]:
        """
        Crée un plan d'action à partir de l'objectif et de la réponse du LLM.
        
        Args:
            objective: L'objectif/question de l'utilisateur
            llm_response: La réponse du LLM
        
        Returns:
            Plan structuré avec tool, params, et fallback_used
        """
        # 1. Essayer de parser un JSON dans la réponse du LLM
        plan = self._try_parse_json(llm_response)
        if plan:
            plan["fallback_used"] = False
            return plan
        
        # 2. Fallback : détecter l'outil par mots-clés
        plan = self._detect_tool_from_text(objective, llm_response)
        plan["fallback_used"] = True
        return plan
    
    def _try_parse_json(self, text: str) -> Optional[Dict]:
        """Tente d'extraire un plan JSON de la réponse du LLM."""
        # Chercher un bloc JSON
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"tool"[^{}]*\}',
            r'\{[^{}]*"action"[^{}]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        # Normaliser
                        tool = data.get("tool") or data.get("action") or data.get("name")
                        params = data.get("params") or data.get("parameters") or data.get("args") or {}
                        
                        if tool:
                            return {
                                "tool": tool,
                                "params": params,
                                "reasoning": data.get("reasoning", "Extrait du JSON LLM")
                            }
                except (json.JSONDecodeError, TypeError):
                    continue
        
        return None
    
    def _detect_tool_from_text(self, objective: str, llm_response: str) -> Dict:
        """Détecte l'outil approprié par analyse de texte."""
        combined_text = f"{objective} {llm_response}".lower()
        objective_lower = objective.lower()
        
        # === PRIORITÉ 1 : Détection directe par le texte de l'objectif ===
        
        # Suppression / destruction → BLOQUER
        if re.search(r'\b(supprime|supprimer|efface|effacer|delete|remove|rm\s|del\s)\b', objective_lower):
            return {
                "tool": "conversation",
                "params": {"response": "🚫 Je ne suis pas autorisée à supprimer des fichiers. C'est une restriction de sécurité."},
                "reasoning": "Demande de suppression détectée → refus de sécurité"
            }
        
        # Code Python explicite
        if re.search(r'(ex[eé]cute.*code|code\s*python|print\s*\(|def\s+\w+\s*\(|import\s+\w+)', objective_lower):
            # Extraire le code
            code = objective
            # Retirer le préfixe "Exécute ce code Python :"
            code = re.sub(r'^.*?:\s*', '', code, count=1)
            if not code.strip() or code.strip() == objective.strip():
                # Chercher du code dans le message
                code_match = re.search(r'(print\s*\(.*\)|def\s+.*|import\s+.*|for\s+.*)', objective)
                if code_match:
                    code = code_match.group(0)
            return {
                "tool": "code_executor",
                "params": {"code": code.strip()},
                "reasoning": "Code Python détecté dans l'objectif"
            }
        
        # Ouvrir un logiciel (universel)
        open_match = re.search(
            r'\b(ouvr[ei]|lance|d[eé]marre|start|open|démarre|demarre)\b[\s\-]*(moi\s+)?(le\s+|la\s+|l\')?(.+)',
            objective_lower
        )
        if open_match:
            raw_apps = open_match.group(4).strip()
            raw_apps = re.sub(r'\s+(et|and|puis|also)\s+(le\s+|la\s+|l\')?', '|', raw_apps)
            app_names = [a.strip() for a in raw_apps.split('|') if a.strip()]
            
            if app_names:
                return {
                    "tool": "app_launcher",
                    "params": {"app_names": app_names},
                    "reasoning": f"Ouverture de logiciel(s) détectée : {', '.join(app_names)}"
                }
                
        # Ping / commande shell explicite
        if re.search(r'(ping\s+\S|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower):
            cmd_match = re.search(r'(ping\s+[\w\.\-]+|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower)
            command = cmd_match.group(0) if cmd_match else "echo commande non détectée"
            return {
                "tool": "shell_command",
                "params": {"command": command},
                "reasoning": "Commande shell détectée dans l'objectif"
            }
        
        # MT5 connexion explicite
        if re.search(r'connect.*mt5|mt5.*connect|connecte.*mt5|connecte-toi.*mt5|metatrader.*connect', objective_lower):
            return {
                "tool": "mt5_tool",
                "params": {"action": "connect"},
                "reasoning": "Connexion MT5 demandée"
            }
        
        # Décomposition de problème explicite
        if re.search(r'd[eé]compos|[eé]tape\s*par\s*[eé]tape|planifi.*action|fais.*plan|comment\s+proc[eé]der', objective_lower):
            return {
                "tool": "task_planner",
                "params": {"problem": objective},
                "reasoning": "Décomposition de problème demandée"
            }
        
        # Recherche web / Wikipedia explicite
        if re.search(r'recherch.*sur|cherch.*info|wikipedia|actualit[eé]|news\s+sur|derni[eè]re.*nouvelle', objective_lower):
            query = objective
            for word in ["recherche", "cherche", "trouve", "sur", "info", "information",
                         "actualité", "news", "qu'est-ce que", "c'est quoi", "des", "les", "de", "du", "la", "le"]:
                query = re.sub(rf'\b{word}\b', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s+', ' ', query).strip()
            if not query:
                query = objective
            
            source = "all"
            if re.search(r'wikipedia|wiki', objective_lower):
                source = "wikipedia"
            elif re.search(r'news|actualit|nouvelle', objective_lower):
                source = "news"
            
            return {
                "tool": "web_search",
                "params": {"query": query, "source": source},
                "reasoning": "Recherche web détectée"
            }
        
        # === PRIORITÉ 2 : Score par patterns (comme avant) ===
        scores = {}
        for tool_name, patterns in self.TOOL_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[tool_name] = score
        
        if not scores:
            return {
                "tool": "conversation",
                "params": {"response": llm_response},
                "reasoning": "Aucun outil détecté — réponse conversationnelle"
            }
        
        best_tool = max(scores, key=scores.get)
        params = self._build_params(best_tool, objective, llm_response)
        
        return {
            "tool": best_tool,
            "params": params,
            "reasoning": f"Outil '{best_tool}' détecté par analyse de texte (score: {scores[best_tool]})"
        }
        
    def _build_params(self, tool: str, objective: str, llm_response: str) -> Dict:
        """Construit les paramètres pour un outil détecté."""
        text = f"{objective} {llm_response}"
        
        if tool == "calculator":
            # Extraire l'expression mathématique
            expr_match = re.search(r'[\d\.\+\-\*/\(\)\^sqrt\s]{3,}', objective)
            if expr_match:
                return {"expression": expr_match.group().strip()}
            # Essayer de trouver dans la réponse
            expr_match = re.search(r'[\d\.\+\-\*/\(\)\^sqrt\s]{3,}', llm_response)
            if expr_match:
                return {"expression": expr_match.group().strip()}
            return {"expression": objective}
        
        elif tool == "datetime":
            return {"format": "%Y-%m-%d %H:%M:%S"}
        
        elif tool == "system_info":
            return {}
        
        elif tool in ("list_directory", "read_file", "write_file", "file_info"):
            # Extraire un chemin
            path_match = re.search(r'[A-Za-z]:\\[^\s"\']+|/[^\s"\']+|\.[\\/][^\s"\']+', text)
            if path_match:
                return {"path": path_match.group()}
            return {"path": "."}
        
        elif tool in ("memory_search", "memory_stats"):
            return {"query": objective}
        
        elif tool == "list_allowed_paths":
            return {}
        
        # Phase 5 outils
        elif tool == "code_executor":
            # Essayer d'extraire du code Python de la réponse LLM
            code_match = re.search(r'```python\s*(.*?)\s*```', llm_response, re.DOTALL)
            if code_match:
                return {"code": code_match.group(1)}
            code_match = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
            if code_match:
                return {"code": code_match.group(1)}
            return {"code": f"# Problème : {objective}\nprint('Solution à implémenter')"}
        
        elif tool == "web_search":
            # Extraire les termes de recherche
            # Retirer les mots déclencheurs
            query = objective
            for word in ["recherche", "cherche", "trouve", "sur", "info", "information",
                         "actualité", "news", "qu'est-ce que", "c'est quoi"]:
                query = re.sub(rf'\b{word}\b', '', query, flags=re.IGNORECASE)
            query = query.strip()
            if not query:
                query = objective
            
            # Détecter la source
            source = "all"
            if re.search(r'wikipedia|wiki', text, re.IGNORECASE):
                source = "wikipedia"
            elif re.search(r'news|actualit|nouvelle', text, re.IGNORECASE):
                source = "news"
            
            return {"query": query, "source": source}
        
        elif tool == "shell_command":
            # Extraire la commande
            cmd_match = re.search(r'`([^`]+)`', text)
            if cmd_match:
                return {"command": cmd_match.group(1)}
            # Essayer de détecter une commande connue
            for cmd in ["ping", "ipconfig", "systeminfo", "tasklist", "hostname",
                        "whoami", "dir", "git", "pip", "python"]:
                if cmd in text.lower():
                    # Extraire la commande complète
                    cmd_match = re.search(rf'({cmd}[^\n.,;]*)', text, re.IGNORECASE)
                    if cmd_match:
                        return {"command": cmd_match.group(1).strip()}
            return {"command": "echo Commande non détectée"}
        
        elif tool == "mt5_tool":
            # Déterminer l'action MT5
            text_lower = text.lower()
            
            if any(w in text_lower for w in ["connect", "connexion", "connecte"]):
                return {"action": "connect"}
            elif any(w in text_lower for w in ["position", "trade ouvert"]):
                return {"action": "positions"}
            elif any(w in text_lower for w in ["compte", "account", "solde", "balance"]):
                return {"action": "account_info"}
            elif any(w in text_lower for w in ["historique", "historical", "donnée", "bougie", "candle"]):
                # Extraire le symbole
                symbol_match = re.search(r'\b([A-Z]{6})\b', text)
                symbol = symbol_match.group(1) if symbol_match else "EURUSD"
                # Extraire le timeframe
                tf_match = re.search(r'\b(M1|M5|M15|M30|H1|H4|D1|W1|MN1)\b', text, re.IGNORECASE)
                tf = tf_match.group(1).upper() if tf_match else "H1"
                return {"action": "historical_data", "symbol": symbol, "timeframe": tf}
            elif any(w in text_lower for w in ["symbole", "symbol", "prix", "cours"]):
                symbol_match = re.search(r'\b([A-Z]{6})\b', text)
                symbol = symbol_match.group(1) if symbol_match else "EURUSD"
                return {"action": "symbol_info", "symbol": symbol}
            else:
                return {"action": "connect"}
        
        elif tool == "task_planner":
            return {"problem": objective}
        
        return {}


# Instance globale
planner = Planner()