"""
Planner — Planifie les actions de l'agent.
Emplacement : micheline/core/planner.py
FICHIER MODIFIÉ — Phase 5 + Phase 6 (Trading Engine + Multi-actions)
"""

import json
import re
from typing import Dict, Any, Optional, List


class Planner:
    """Planifie les actions en fonction de l'objectif et de la réponse du LLM."""

    TOOL_PATTERNS = {
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
            r"connect.*mt5", r"mt5.*connect", r"connecte.*mt5",
            r"position.*mt5", r"compte.*mt5", r"solde.*mt5",
            r"symbole.*mt5", r"prix.*mt5", r"cours.*mt5",
            r"donn[eé]es.*mt5", r"bougie.*mt5", r"candle.*mt5"
        ],
        "trading_quick_test": [
            r"test.*rapide.*trading", r"teste.*strat[eé]gi",
            r"quick.*test.*trad", r"essai.*trading",
            r"test.*trading", r"trading.*test"
        ],
        "trading_search": [
            r"cherch.*strat[eé]gi.*trading", r"optimis.*strat[eé]gi",
            r"trouv.*strat[eé]gi", r"recherch.*strat[eé]gi",
            r"meilleur.*strat[eé]gi", r"search.*strat",
            r"optimis.*trading", r"cherch.*trading"
        ],
        "trading_improve": [
            r"am[eé]lior.*strat[eé]gi", r"improve.*strat",
            r"optimis.*strat[eé]gi.*exist", r"mutation.*strat"
        ],
        "trading_report": [
            r"rapport.*trading", r"r[eé]sum[eé].*trading",
            r"bilan.*trading", r"report.*trading",
            r"session.*trading", r"stat.*trading"
        ],
        "trading_top_strategies": [
            r"top.*strat[eé]gi", r"meilleur.*strat[eé]gi",
            r"classement.*strat", r"best.*strat"
        ],
        "task_planner": [
            r"d[eé]compos.*probl[eè]me", r"plan.*action", r"[eé]tape.*par.*[eé]tape",
            r"comment.*proc[eé]der", r"planifi", r"fais.*plan",
            r"organis.*t[aâ]che", r"projet.*complexe"
        ],
    }

    def create_plan(self, objective: str, llm_response: str) -> Dict[str, Any]:
        plan = self._try_parse_json(llm_response)
        if plan:
            plan["fallback_used"] = False
            return plan

        plan = self._detect_tool_from_text(objective, llm_response)
        plan["fallback_used"] = True
        return plan

    def _try_parse_json(self, text: str) -> Optional[Dict]:
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
        combined_text = f"{objective} {llm_response}".lower()
        objective_lower = objective.lower()

        # ═══════════════════════════════════════
        # PRIORITÉ 0 : Trading Engine
        # ═══════════════════════════════════════

        trading_result = self._detect_trading(objective_lower)
        if trading_result:
            return trading_result

        # ═══════════════════════════════════════
        # PRIORITÉ 1 : Détection directe
        # ═══════════════════════════════════════

        # Suppression → BLOQUER
        if re.search(r'\b(supprime|supprimer|efface|effacer|delete|remove|rm\s|del\s)\b', objective_lower):
            return {
                "tool": "conversation",
                "params": {"response": "🚫 Je ne suis pas autorisée à supprimer des fichiers."},
                "reasoning": "Demande de suppression → refus"
            }

        # Code Python explicite
        if re.search(r'(ex[eé]cute.*code|code\s*python|print\s*\(|def\s+\w+\s*\(|import\s+\w+)', objective_lower):
            code = objective
            code = re.sub(r'^.*?:\s*', '', code, count=1)
            if not code.strip() or code.strip() == objective.strip():
                code_match = re.search(r'(print\s*\(.*\)|def\s+.*|import\s+.*|for\s+.*)', objective)
                if code_match:
                    code = code_match.group(0)
            return {
                "tool": "code_executor",
                "params": {"code": code.strip()},
                "reasoning": "Code Python détecté"
            }

        # Ouvrir un logiciel
        open_match = re.search(
            r'\b(ouvr[ei]|lance|d[eé]marre|start|open|démarre|demarre)\b[\s\-]*(moi\s+)?(le\s+|la\s+|l\')?(.+)',
            objective_lower
        )
        if open_match:
            raw_apps = open_match.group(4).strip()
            trading_keywords = [
                "trading", "backtest", "stratégie", "strategie", "strat",
                "forex", "bourse", "mt5", "metatrader", "optimis"
            ]
            is_trading_context = any(kw in raw_apps for kw in trading_keywords)

            if not is_trading_context:
                raw_apps = re.sub(r'\s+(et|and|puis|also)\s+(le\s+|la\s+|l\')?', '|', raw_apps)
                app_names = [a.strip() for a in raw_apps.split('|') if a.strip()]
                if app_names:
                    return {
                        "tool": "app_launcher",
                        "params": {"app_names": app_names},
                        "reasoning": f"Ouverture : {', '.join(app_names)}"
                    }

        # Ping / shell
        if re.search(r'(ping\s+\S|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower):
            cmd_match = re.search(r'(ping\s+[\w\.\-]+|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower)
            command = cmd_match.group(0) if cmd_match else "echo commande non détectée"
            return {
                "tool": "shell_command",
                "params": {"command": command},
                "reasoning": "Commande shell détectée"
            }

        # MT5 connexion
        if re.search(r'connect.*mt5|mt5.*connect|connecte.*mt5|connecte-toi.*mt5|metatrader.*connect', objective_lower):
            return {
                "tool": "mt5_tool",
                "params": {"action": "connect"},
                "reasoning": "Connexion MT5 demandée"
            }

        # Décomposition
        if re.search(r'd[eé]compos|[eé]tape\s*par\s*[eé]tape|planifi.*action|fais.*plan|comment\s+proc[eé]der', objective_lower):
            return {
                "tool": "task_planner",
                "params": {"problem": objective},
                "reasoning": "Décomposition demandée"
            }

        # Recherche web
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

        # ═══════════════════════════════════════
        # PRIORITÉ 2 : Score par patterns
        # ═══════════════════════════════════════
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
                "reasoning": "Aucun outil détecté — conversation"
            }

        best_tool = max(scores, key=scores.get)
        params = self._build_params(best_tool, objective, llm_response)

        return {
            "tool": best_tool,
            "params": params,
            "reasoning": f"'{best_tool}' détecté (score: {scores[best_tool]})"
        }

    # ═══════════════════════════════════════
    # DÉTECTION TRADING
    # ═══════════════════════════════════════

    def _detect_trading(self, objective_lower: str) -> Optional[Dict]:
        has_trading_word = bool(re.search(
            r'trading|backtest|strat[eé]gi|forex|bourse',
            objective_lower
        ))

        has_symbol = bool(re.search(
            r'\b(eurusd|gbpusd|usdjpy|usdchf|audusd|usdcad|nzdusd'
            r'|eurjpy|gbpjpy|eurgbp|eurcad|eurchf|gbpcad|gbpchf'
            r'|cadjpy|chfjpy|cadchf|xauusd|xagusd'
            r'|usa500|usaind|usatec|ger40|uk100|fra40|jp225)\b',
            objective_lower
        ))

        has_timeframe = bool(re.search(
            r'\b(m1|m5|m15|m30|h1|h4|d1)\b',
            objective_lower
        ))

        trading_context = (
            has_trading_word
            or (has_symbol and has_timeframe)
            or (has_symbol and bool(re.search(
                r'strat[eé]gi|backtest|optimis|test|cherch|trouv|am[eé]lior|rapport|r[eé]sum',
                objective_lower
            )))
        )

        if not trading_context:
            return None

        # Info stratégie
        if re.search(
            r"c.est\s+quoi.*strat|quelle.*strat|montre.*strat"
            r"|affiche.*strat|d[eé]tail.*strat|ta\s+strat"
            r"|strat[eé]gi.*sur\s+\w{6}|donne.*strat",
            objective_lower
        ):
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Recherche stratégie spécifique"
            }

        # Test rapide
        if re.search(
            r'test.*rapide|quick.*test|essai.*rapide|rapide.*test'
            r'|teste.*quelques|test.*strat[eé]gi'
            r'|lance.*test.*trad|fais.*test.*trad',
            objective_lower
        ):
            count = self._extract_number(objective_lower, default=5)
            return {
                "tool": "trading_quick_test",
                "params": {"count": count},
                "reasoning": f"Test rapide {count} stratégies"
            }

        # Recherche / Optimisation / Trouve / Fait
        if re.search(
            r'cherch.*strat|trouv.*strat|optimis.*strat'
            r'|recherch.*strat|meilleur.*strat|search.*strat'
            r'|optimis.*trading|cherch.*trading|lance.*recherch.*trad'
            r'|lance.*optimis|trouv.*trading|trouve.*trad'
            r'|fait.*strat.*trad|fais.*strat.*trad'
            r'|fait.*trading.*rentable|fais.*trading.*rentable'
            r'|strat[eé]gi.*rentable',
            objective_lower
        ):
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 5)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Recherche stratégie optimale"
            }

        # Amélioration
        if re.search(
            r'am[eé]lior.*strat|improve.*strat|optimis.*exist'
            r'|mutation.*strat|affin.*strat',
            objective_lower
        ):
            return {
                "tool": "trading_improve",
                "params": {"iterations": 20, "mutation_strength": 0.2},
                "reasoning": "Amélioration stratégie"
            }

        # Rapport
        if re.search(
            r'rapport.*trad|bilan.*trad|r[eé]sum[eé].*trad'
            r'|report.*trad|stat.*trad|session.*trad',
            objective_lower
        ):
            return {
                "tool": "trading_report",
                "params": {},
                "reasoning": "Rapport trading"
            }

        # Top stratégies
        if re.search(
            r'top.*strat|meilleur.*strat|classement.*strat'
            r'|best.*strat|hall.*fame',
            objective_lower
        ):
            count = self._extract_number(objective_lower, default=5)
            return {
                "tool": "trading_top_strategies",
                "params": {"count": count},
                "reasoning": f"Top {count} stratégies"
            }

        # Backtest
        if re.search(r'backtest|back.test', objective_lower):
            params = self._extract_trading_search_params(objective_lower)
            params["count"] = params.pop("population_size", 5)
            return {
                "tool": "trading_quick_test",
                "params": params,
                "reasoning": "Backtest → test rapide"
            }

        # Symbole détecté
        if has_symbol:
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Trading avec symbole détecté"
            }

        # Rapide
        if re.search(r'rapide|vite|quick|fast', objective_lower):
            return {
                "tool": "trading_quick_test",
                "params": {"count": 5},
                "reasoning": "Trading rapide → test rapide"
            }

        return {
            "tool": "trading_search",
            "params": {"population_size": 10, "max_generations": 3},
            "reasoning": "Trading générique → recherche légère"
        }

    def _extract_trading_search_params(self, text: str) -> Dict[str, Any]:
        params = {}

        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCHF",
                    "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY",
                    "EURGBP", "XAGUSD"]
        for symbol in symbols:
            if symbol.lower() in text:
                params["symbols"] = [symbol]
                break

        timeframes = {"m1": "M1", "m5": "M5", "m15": "M15", "m30": "M30",
                      "h1": "H1", "h4": "H4", "d1": "D1"}
        for key, value in timeframes.items():
            if key in text:
                params["timeframes"] = [value]
                break

        gen_match = re.search(r'(\d+)\s*(?:gen|génération|generation|iter)', text)
        if gen_match:
            params["max_generations"] = int(gen_match.group(1))

        pop_match = re.search(r'(\d+)\s*(?:pop|population|strat)', text)
        if pop_match:
            params["population_size"] = int(pop_match.group(1))

        return params

    def _extract_number(self, text: str, default: int = 5) -> int:
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else default

    # ═══════════════════════════════════════
    # BUILD PARAMS
    # ═══════════════════════════════════════

    def _build_params(self, tool: str, objective: str, llm_response: str) -> Dict:
        text = f"{objective} {llm_response}"

        if tool == "calculator":
            expr_match = re.search(r'[\d\.\+\-\*/\(\)\^sqrt\s]{3,}', objective)
            if expr_match:
                return {"expression": expr_match.group().strip()}
            return {"expression": objective}

        elif tool == "datetime":
            return {"format": "%Y-%m-%d %H:%M:%S"}

        elif tool == "system_info":
            return {}

        elif tool in ("list_directory", "read_file", "write_file", "file_info"):
            path_match = re.search(r'[A-Za-z]:\\[^\s"\']+|/[^\s"\']+|\.[\\/][^\s"\']+', text)
            if path_match:
                return {"path": path_match.group()}
            return {"path": "."}

        elif tool in ("memory_search", "memory_stats"):
            return {"query": objective}

        elif tool == "list_allowed_paths":
            return {}

        elif tool == "code_executor":
            code_match = re.search(r'```python\s*(.*?)\s*```', llm_response, re.DOTALL)
            if code_match:
                return {"code": code_match.group(1)}
            code_match = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
            if code_match:
                return {"code": code_match.group(1)}
            return {"code": f"# {objective}\nprint('À implémenter')"}

        elif tool == "web_search":
            query = objective
            for word in ["recherche", "cherche", "trouve", "sur", "info",
                         "actualité", "news", "qu'est-ce que", "c'est quoi"]:
                query = re.sub(rf'\b{word}\b', '', query, flags=re.IGNORECASE)
            query = query.strip() or objective
            source = "all"
            if re.search(r'wikipedia|wiki', text, re.IGNORECASE):
                source = "wikipedia"
            elif re.search(r'news|actualit|nouvelle', text, re.IGNORECASE):
                source = "news"
            return {"query": query, "source": source}

        elif tool == "shell_command":
            cmd_match = re.search(r'`([^`]+)`', text)
            if cmd_match:
                return {"command": cmd_match.group(1)}
            for cmd in ["ping", "ipconfig", "systeminfo", "tasklist",
                        "hostname", "whoami", "dir", "git", "pip", "python"]:
                if cmd in text.lower():
                    cmd_match = re.search(rf'({cmd}[^\n.,;]*)', text, re.IGNORECASE)
                    if cmd_match:
                        return {"command": cmd_match.group(1).strip()}
            return {"command": "echo Commande non détectée"}

        elif tool == "mt5_tool":
            text_lower = text.lower()
            if any(w in text_lower for w in ["connect", "connexion", "connecte"]):
                return {"action": "connect"}
            elif any(w in text_lower for w in ["position", "trade ouvert"]):
                return {"action": "positions"}
            elif any(w in text_lower for w in ["compte", "account", "solde", "balance"]):
                return {"action": "account_info"}
            elif any(w in text_lower for w in ["historique", "donnée", "bougie"]):
                symbol_match = re.search(r'\b([A-Z]{6})\b', text)
                symbol = symbol_match.group(1) if symbol_match else "EURUSD"
                return {"action": "historical_data", "symbol": symbol}
            elif any(w in text_lower for w in ["symbole", "prix", "cours"]):
                symbol_match = re.search(r'\b([A-Z]{6})\b', text)
                symbol = symbol_match.group(1) if symbol_match else "EURUSD"
                return {"action": "symbol_info", "symbol": symbol}
            return {"action": "connect"}

        elif tool == "task_planner":
            return {"problem": objective}

        elif tool == "trading_quick_test":
            return {"count": self._extract_number(objective, default=5)}

        elif tool == "trading_search":
            params = self._extract_trading_search_params(objective.lower())
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 5)
            return params

        elif tool == "trading_improve":
            return {"iterations": 20, "mutation_strength": 0.2}

        elif tool == "trading_report":
            return {}

        elif tool == "trading_top_strategies":
            return {"count": self._extract_number(objective, default=5)}

        return {}


planner = Planner()