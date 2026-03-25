"""
micheline/core/planner.py

Planificateur intelligent avec :
- Détection multi-actions (split "et/puis/ensuite")
- Détection trading avancée
- Détection app_launcher (ouvre paint, notepad, etc.)
- Gestion des échecs et fallbacks
- Attributs llm_client et set_tools_description() pour main.py
"""

import json
import re
import logging
import unicodedata
from typing import Dict, Any, List, Optional

logger = logging.getLogger("micheline.planner")


class Planner:

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

    def __init__(self, available_tools: List[str] = None):
        self.available_tools = available_tools or []
        self.failure_history: List[Dict[str, Any]] = []
        self.attempt_count: Dict[str, int] = {}

        # Attributs attendus par main.py
        self.llm_client = None
        self._tools_description: str = ""

        logger.info(f"Planner initialisé avec {len(self.available_tools)} outils")

    def set_tools_description(self, description: str):
        """Appelé par main.py pour donner au planner la description des outils."""
        self._tools_description = description or ""

    def update_tools(self, tools: List[str]):
        self.available_tools = tools

    def record_failure(self, tool_name: str, params: dict = None, error: str = ""):
        self.failure_history.append({"tool": tool_name, "params": params, "error": error})
        self.attempt_count[tool_name] = self.attempt_count.get(tool_name, 0) + 1

    def record_success(self, tool_name: str):
        self.attempt_count[tool_name] = 0

    def reset(self):
        self.failure_history.clear()
        self.attempt_count.clear()

    # ═══════════════════════════════════════
    # SPLIT MULTI-ACTIONS
    # ═══════════════════════════════════════

    def split_objectives(self, objective: str) -> List[str]:
        """
        Détecte si le message contient plusieurs actions distinctes.
        Ex: "fait moi une stratégie EURUSD et ouvre moi paint"
        → ["fait moi une stratégie EURUSD", "ouvre moi paint"]
        """
        text = objective.strip()

        action_words = (
            r'ouvr[ei]|lance|d[eé]marre|start|open'
            r'|ferm[eé]|ferme'
            r'|cherch[eé]|trouv[eé]|recherch'
            r'|fais|fait'
            r'|cr[eé]e|supprim'
            r'|affich[eé]|montr[eé]'
            r'|calcul[eé]'
            r'|ex[eé]cute'
            r'|connecte'
            r'|teste'
        )

        split_pattern = (
            rf'\s+(?:et|puis|ensuite|aussi|également)\s+'
            rf'(?:(?:moi|le|la|les|l\'|un|une|des|du)\s+)*'
            rf'(?={action_words})'
        )

        parts = re.split(split_pattern, text, flags=re.IGNORECASE)

        cleaned = []
        for p in parts:
            p = p.strip().rstrip('.!?')
            if len(p) > 3:
                cleaned.append(p)

        if len(cleaned) > 1:
            logger.info(f"Multi-actions détectées: {cleaned}")
            return cleaned

        return [text]

    # ═══════════════════════════════════════
    # PLAN PRINCIPAL
    # ═══════════════════════════════════════

    def plan(self, objective: str, context: dict = None, llm=None) -> Dict[str, Any]:
        """
        Planifie l'action pour UN SEUL objectif (déjà splitté).
        """
        context = context or {}
        objective_lower = objective.lower()

        def remove_accents(s):
            try:
                return "".join(
                    c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn"
                )
            except Exception:
                return s

        text_normalized = remove_accents(objective_lower)

        # ═══════════════════════════════
        # PRIORITÉ 0 : TRADING
        # ═══════════════════════════════
        trading_result = self._detect_trading(objective_lower)
        if trading_result:
            # Vérifier les échecs pour adapter
            ts_failures = self.attempt_count.get("trading_search", 0)
            tg_failures = self.attempt_count.get("trading_generate", 0)

            tool = trading_result.get("tool", "trading_search")

            if tool == "trading_search" and ts_failures >= 2 and tg_failures == 0:
                return {
                    "tool": "trading_generate",
                    "params": trading_result.get("params", {}),
                    "reasoning": f"Fallback trading_generate (trading_search échoué {ts_failures}x)",
                    "fallback": "llm_direct"
                }
            elif tool == "trading_search" and ts_failures >= 1:
                params = trading_result.get("params", {})
                params["population_size"] = min(params.get("population_size", 10), 5)
                params["max_generations"] = min(params.get("max_generations", 3), 1)
                return {
                    "tool": "trading_search",
                    "params": params,
                    "reasoning": "Retry trading_search avec params réduits",
                    "fallback": "trading_generate"
                }

            trading_result["fallback"] = "trading_generate"
            return trading_result

        # ═══════════════════════════════
        # PRIORITÉ 1 : APP LAUNCHER
        # ═══════════════════════════════
        open_match = re.search(
            r'\b(ouvr[ei]|lance|d[eé]marre|start|open|démarre|demarre)\b'
            r'[\s\-]*(moi\s+)?(le\s+|la\s+|l\')?(.+)',
            objective_lower
        )
        if open_match:
            raw_apps = open_match.group(4).strip()

            # Vérifier que ce n'est PAS du trading
            trading_kw = [
                "trading", "backtest", "stratégie", "strategie", "strat",
                "forex", "bourse", "mt5", "metatrader", "optimis"
            ]
            is_trading = any(kw in raw_apps for kw in trading_kw)

            if not is_trading:
                raw_apps = re.sub(
                    r'\s+(et|and|puis|also)\s+(le\s+|la\s+|l\')?',
                    '|', raw_apps
                )
                app_names = [a.strip() for a in raw_apps.split('|') if a.strip()]
                if app_names:
                    return {
                        "tool": "app_launcher",
                        "params": {"app_names": app_names},
                        "reasoning": f"Ouverture application(s): {', '.join(app_names)}",
                        "fallback": None
                    }

        # ═══════════════════════════════
        # PRIORITÉ 2 : SUPPRESSION → BLOQUER
        # ═══════════════════════════════
        if re.search(r'\b(supprime|supprimer|efface|effacer|delete|remove|rm\s|del\s)\b', objective_lower):
            return {
                "tool": "conversation",
                "params": {"response": "🚫 Je ne suis pas autorisée à supprimer des fichiers."},
                "reasoning": "Demande de suppression → refus",
                "fallback": None
            }

        # ═══════════════════════════════
        # PRIORITÉ 3 : CODE PYTHON
        # ═══════════════════════════════
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
                "reasoning": "Code Python détecté",
                "fallback": None
            }

        # ═══════════════════════════════
        # PRIORITÉ 4 : PING / SHELL
        # ═══════════════════════════════
        if re.search(r'(ping\s+\S|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower):
            cmd_match = re.search(r'(ping\s+[\w\.\-]+|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower)
            command = cmd_match.group(0) if cmd_match else "echo commande non détectée"
            return {
                "tool": "shell_command",
                "params": {"command": command},
                "reasoning": "Commande shell détectée",
                "fallback": None
            }

        # ═══════════════════════════════
        # PRIORITÉ 5 : MT5
        # ═══════════════════════════════
        if re.search(r'connect.*mt5|mt5.*connect|connecte.*mt5|metatrader.*connect', objective_lower):
            return {
                "tool": "mt5_tool",
                "params": {"action": "connect"},
                "reasoning": "Connexion MT5 demandée",
                "fallback": None
            }

        # ═══════════════════════════════
        # PRIORITÉ 6 : DÉCOMPOSITION
        # ═══════════════════════════════
        if re.search(r'd[eé]compos|[eé]tape\s*par\s*[eé]tape|planifi.*action|fais.*plan', objective_lower):
            return {
                "tool": "task_planner",
                "params": {"problem": objective},
                "reasoning": "Décomposition demandée",
                "fallback": None
            }

        # ═══════════════════════════════
        # PRIORITÉ 7 : RECHERCHE WEB
        # ═══════════════════════════════
        if re.search(r'recherch.*sur|cherch.*info|wikipedia|actualit|news\s+sur', objective_lower):
            query = objective
            for word in ["recherche", "cherche", "trouve", "sur", "info",
                         "actualité", "news", "qu'est-ce que", "c'est quoi"]:
                query = re.sub(rf'\b{word}\b', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s+', ' ', query).strip() or objective
            return {
                "tool": "web_search",
                "params": {"query": query},
                "reasoning": "Recherche web détectée",
                "fallback": "llm_direct"
            }

        # ═══════════════════════════════
        # PRIORITÉ 8 : SCORE PAR PATTERNS
        # ═══════════════════════════════
        combined_text = f"{objective}".lower()
        scores = {}
        for tool_name, patterns in self.TOOL_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, combined_text, re.IGNORECASE))
            if score > 0:
                scores[tool_name] = score

        if scores:
            best_tool = max(scores, key=scores.get)
            params = self._build_params(best_tool, objective)
            return {
                "tool": best_tool,
                "params": params,
                "reasoning": f"'{best_tool}' détecté (score: {scores[best_tool]})",
                "fallback": None
            }

        # ═══════════════════════════════
        # FALLBACK : LLM DIRECT
        # ═══════════════════════════════
        effective_llm = llm or self.llm_client
        if effective_llm:
            return {
                "tool": "llm_direct",
                "params": {"prompt": objective},
                "reasoning": "Aucun outil détecté → LLM direct",
                "fallback": None
            }

        return {
            "tool": "none",
            "params": {},
            "reasoning": "Aucun outil trouvé",
            "fallback": None
        }

    # ═══════════════════════════════════════
    # DÉTECTION TRADING
    # ═══════════════════════════════════════

    def _detect_trading(self, objective_lower: str) -> Optional[Dict]:
        has_trading_word = bool(re.search(
            r'trading|backtest|strat[eé]gi|forex|bourse', objective_lower
        ))
        has_symbol = bool(re.search(
            r'\b(eurusd|gbpusd|usdjpy|usdchf|audusd|usdcad|nzdusd'
            r'|eurjpy|gbpjpy|eurgbp|eurcad|eurchf|gbpcad|gbpchf'
            r'|cadjpy|chfjpy|cadchf|xauusd|xagusd)\b',
            objective_lower
        ))
        has_timeframe = bool(re.search(r'\b(m1|m5|m15|m30|h1|h4|d1)\b', objective_lower))

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

        # Recherche / Optimisation / Trouve / Fait
        if re.search(
            r'cherch.*strat|trouv.*strat|optimis.*strat'
            r'|recherch.*strat|meilleur.*strat|search.*strat'
            r'|optimis.*trading|cherch.*trading|lance.*recherch.*trad'
            r'|lance.*optimis|trouv.*trading|trouve.*trad'
            r'|fait.*strat.*trad|fais.*strat.*trad'
            r'|fait.*trading.*rentable|fais.*trading.*rentable'
            r'|strat[eé]gi.*rentable'
            r'|strat[eé]gi.*sur\s+\w{6}|donne.*strat',
            objective_lower
        ):
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Recherche stratégie spécifique",
                "fallback": "trading_generate"
            }

        # Test rapide
        if re.search(r'test.*rapide|quick.*test|essai.*rapide|test.*strat', objective_lower):
            count = self._extract_number(objective_lower, default=5)
            return {
                "tool": "trading_quick_test",
                "params": {"count": count},
                "reasoning": f"Test rapide {count} stratégies",
                "fallback": None
            }

        # Amélioration
        if re.search(r'am[eé]lior.*strat|improve.*strat|optimis.*exist', objective_lower):
            return {
                "tool": "trading_improve",
                "params": {"iterations": 20, "mutation_strength": 0.2},
                "reasoning": "Amélioration stratégie",
                "fallback": None
            }

        # Rapport
        if re.search(r'rapport.*trad|bilan.*trad|r[eé]sum[eé].*trad|stat.*trad', objective_lower):
            return {
                "tool": "trading_report",
                "params": {},
                "reasoning": "Rapport trading",
                "fallback": None
            }

        # Top stratégies
        if re.search(r'top.*strat|classement.*strat|best.*strat', objective_lower):
            count = self._extract_number(objective_lower, default=5)
            return {
                "tool": "trading_top_strategies",
                "params": {"count": count},
                "reasoning": f"Top {count} stratégies",
                "fallback": None
            }

        # Backtest
        if re.search(r'backtest|back.test', objective_lower):
            params = self._extract_trading_search_params(objective_lower)
            params["count"] = params.pop("population_size", 5)
            return {
                "tool": "trading_quick_test",
                "params": params,
                "reasoning": "Backtest → test rapide",
                "fallback": None
            }

        # Symbole détecté seul
        if has_symbol:
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Trading avec symbole détecté",
                "fallback": "trading_generate"
            }

        return {
            "tool": "trading_search",
            "params": {"population_size": 10, "max_generations": 3},
            "reasoning": "Trading générique → recherche",
            "fallback": "trading_generate"
        }

    def _extract_trading_search_params(self, text: str) -> Dict[str, Any]:
        params = {}
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCHF",
                    "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY",
                    "EURGBP", "XAGUSD"]
        for s in symbols:
            if s.lower() in text:
                params["symbols"] = [s]
                break

        tfs = {"m1": "M1", "m5": "M5", "m15": "M15", "m30": "M30",
               "h1": "H1", "h4": "H4", "d1": "D1"}
        for k, v in tfs.items():
            if k in text:
                params["timeframes"] = [v]
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

    def _build_params(self, tool: str, objective: str) -> Dict:
        if tool == "calculator":
            expr_match = re.search(r'[\d\.\+\-\*/\(\)\^sqrt\s]{3,}', objective)
            return {"expression": expr_match.group().strip() if expr_match else objective}
        elif tool == "datetime":
            return {"format": "%Y-%m-%d %H:%M:%S"}
        elif tool == "system_info":
            return {}
        elif tool in ("list_directory", "read_file", "write_file", "file_info"):
            path_match = re.search(r'[A-Za-z]:\\[^\s"\']+|/[^\s"\']+|\.[\\/][^\s"\']+', objective)
            return {"path": path_match.group() if path_match else "."}
        elif tool in ("memory_search", "memory_stats"):
            return {"query": objective}
        elif tool == "list_allowed_paths":
            return {}
        elif tool == "code_executor":
            return {"code": f"# {objective}\nprint('À implémenter')"}
        elif tool == "web_search":
            query = objective
            for w in ["recherche", "cherche", "trouve", "sur", "info"]:
                query = re.sub(rf'\b{w}\b', '', query, flags=re.IGNORECASE)
            return {"query": query.strip() or objective}
        elif tool == "shell_command":
            for cmd in ["ping", "ipconfig", "systeminfo", "tasklist", "hostname", "whoami"]:
                if cmd in objective.lower():
                    cmd_match = re.search(rf'({cmd}[^\n.,;]*)', objective, re.IGNORECASE)
                    if cmd_match:
                        return {"command": cmd_match.group(1).strip()}
            return {"command": "echo Commande non détectée"}
        elif tool == "mt5_tool":
            return {"action": "connect"}
        elif tool == "task_planner":
            return {"problem": objective}
        elif tool == "trading_search":
            params = self._extract_trading_search_params(objective.lower())
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return params
        elif tool == "trading_quick_test":
            return {"count": self._extract_number(objective, default=5)}
        elif tool == "trading_improve":
            return {"iterations": 20, "mutation_strength": 0.2}
        elif tool == "trading_report":
            return {}
        elif tool == "trading_top_strategies":
            return {"count": self._extract_number(objective, default=5)}
        return {}