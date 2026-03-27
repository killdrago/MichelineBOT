"""
Planner — Planifie les actions de l'agent.
Emplacement : micheline/core/planner.py
FICHIER COMPLET — Compatible agent_bridge + agent_loop + main.py
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
        "app_launcher": [
            r"ouvr[ei]", r"lance\b", r"d[eé]marre", r"start\b", r"open\b",
            r"ferm[eé]", r"ferme\b"
        ],
    }

    KNOWN_APPS = [
        "paint", "notepad", "bloc-notes", "bloc notes", "calculatrice",
        "calculator", "word", "excel", "chrome", "firefox", "edge",
        "explorer", "explorateur", "cmd", "terminal", "powershell",
        "vlc", "spotify", "discord", "steam", "vscode", "code",
        "visual studio", "gimp", "photoshop", "obs", "teams",
        "outlook", "thunderbird", "brave", "opera", "safari",
        "winamp", "media player", "lecteur", "snipping", "capture"
    ]

    def __init__(self):
        """Initialise le planner."""
        self.tools_description = ""
        self.llm_client = None
        self.llm = None
        self.last_plan = None
        self.plan_history = []
        self.available_tools = {}
        self.tool_success_count = {}
        self.tool_failure_count = {}
        self.tool_failure_log = []

    # ═══════════════════════════════════════════════════
    # MÉTHODES REQUISES PAR agent_bridge.py ET main.py
    # ═══════════════════════════════════════════════════

    def set_tools_description(self, description: str):
        """
        Reçoit la description des outils disponibles.
        Appelé par main.py.
        """
        self.tools_description = description

    def update_tools(self, tools):
        """
        Met à jour la liste des outils disponibles.
        Appelé par agent_bridge.py ligne 94.
        """
        if isinstance(tools, dict):
            self.available_tools = tools
        elif isinstance(tools, list):
            self.available_tools = {t: {} for t in tools}
        else:
            self.available_tools = {}

    def reset(self):
        """
        Réinitialise l'état du planner.
        Appelé par agent_bridge.py ligne 162.
        """
        self.last_plan = None
        self.plan_history = []
        self.tool_failure_log = []

    def record_success(self, tool_name: str):
        """
        Enregistre qu'un outil a réussi.
        Appelé par agent_bridge.py lignes 227 et 262.
        """
        self.tool_success_count[tool_name] = self.tool_success_count.get(tool_name, 0) + 1

    def record_failure(self, tool_name: str, params: dict = None, error_msg: str = ""):
        """
        Enregistre qu'un outil a échoué.
        Appelé par agent_bridge.py ligne 253.
        """
        self.tool_failure_count[tool_name] = self.tool_failure_count.get(tool_name, 0) + 1
        self.tool_failure_log.append({
            "tool": tool_name,
            "params": params or {},
            "error": error_msg
        })
        if len(self.tool_failure_log) > 100:
            self.tool_failure_log = self.tool_failure_log[-100:]

    def plan(self, objective: str, **kwargs) -> Dict[str, Any]:
        """
        Planifie une action.
        Appelé par agent_bridge.py ligne 177.
        Accepte n'importe quel keyword argument pour compatibilité.
        
        kwargs possibles: context, llm, llm_response, tools, history, etc.
        """
        # Récupérer le LLM depuis les kwargs si fourni
        llm = kwargs.get("llm", None)
        context = kwargs.get("context", None)
        llm_response = kwargs.get("llm_response", "")

        # Si un LLM est passé en argument, l'utiliser pour générer une réponse
        if not llm_response and llm:
            try:
                prompt = f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                if hasattr(llm, 'invoke'):
                    llm_response = llm.invoke(prompt)
                elif hasattr(llm, 'generate'):
                    llm_response = llm.generate(prompt)
                elif hasattr(llm, '__call__'):
                    llm_response = llm(prompt)
                elif hasattr(llm, 'chat'):
                    llm_response = llm.chat(prompt)
                else:
                    llm_response = str(llm)
                if not isinstance(llm_response, str):
                    llm_response = str(llm_response)
            except Exception as e:
                llm_response = ""

        # Si toujours pas de réponse, essayer avec le LLM interne
        if not llm_response and self.llm_client:
            try:
                if hasattr(self.llm_client, 'invoke'):
                    llm_response = self.llm_client.invoke(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                elif hasattr(self.llm_client, 'generate'):
                    llm_response = self.llm_client.generate(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                elif hasattr(self.llm_client, '__call__'):
                    llm_response = self.llm_client(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                if not isinstance(llm_response, str):
                    llm_response = str(llm_response)
            except Exception:
                llm_response = ""

        # Si toujours pas de réponse, essayer avec self.llm
        if not llm_response and self.llm:
            try:
                if hasattr(self.llm, 'invoke'):
                    llm_response = self.llm.invoke(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                elif hasattr(self.llm, 'generate'):
                    llm_response = self.llm.generate(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                elif hasattr(self.llm, '__call__'):
                    llm_response = self.llm(
                        f"L'utilisateur demande: {objective}\nRéponds de manière concise."
                    )
                if not isinstance(llm_response, str):
                    llm_response = str(llm_response)
            except Exception:
                llm_response = ""

        return self.create_plan(objective, llm_response or "")

    def get_last_plan(self) -> Optional[Dict]:
        """Retourne le dernier plan créé."""
        return self.last_plan

    def get_plan_history(self) -> List[Dict]:
        """Retourne l'historique des plans."""
        return self.plan_history

    def get_tool_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation des outils."""
        return {
            "success": dict(self.tool_success_count),
            "failures": dict(self.tool_failure_count),
            "recent_errors": self.tool_failure_log[-10:]
        }

    # ═══════════════════════════════════════════════════
    # SPLIT OBJECTIVES
    # ═══════════════════════════════════════════════════

    def split_objectives(self, objective: str) -> List[str]:
        """
        Détecte si le message contient plusieurs actions distinctes.
        Appelé par agent_bridge.py ligne 115 et agent_loop.py ligne 89.
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
            r'|dis|donne|quel'
        )

        # TENTATIVE 1 : Conjonction + verbe d'action
        split_pattern = (
            r'\s+(?:et|puis|ensuite|aussi|également)\s+'
            r'(?:(?:moi|le|la|les|l\'|un|une|des|du)\s+)*'
            r'(?=' + action_words + r')'
        )
        parts = re.split(split_pattern, text, flags=re.IGNORECASE)
        cleaned = [p.strip().rstrip('.!?') for p in parts if len(p.strip()) > 3]
        if len(cleaned) > 1:
            return cleaned

        # TENTATIVE 2 : Virgule + verbe d'action
        alt_pattern = (
            r',\s*'
            r'(?:(?:moi|le|la|les|l\'|un|une|des|du)\s+)*'
            r'(?=' + action_words + r')'
        )
        parts2 = re.split(alt_pattern, text, flags=re.IGNORECASE)
        cleaned2 = [p.strip().rstrip('.!?') for p in parts2 if len(p.strip()) > 3]
        if len(cleaned2) > 1:
            return cleaned2

        # TENTATIVE 3 : Point + verbe d'action
        alt_pattern2 = (
            r'\.\s+'
            r'(?:(?:moi|le|la|les|l\'|un|une|des|du)\s+)*'
            r'(?=' + action_words + r')'
        )
        parts3 = re.split(alt_pattern2, text, flags=re.IGNORECASE)
        cleaned3 = [p.strip().rstrip('.!?') for p in parts3 if len(p.strip()) > 3]
        if len(cleaned3) > 1:
            return cleaned3

        return [text]

    # ═══════════════════════════════════════════════════
    # CREATE PLAN
    # ═══════════════════════════════════════════════════

    def create_plan(self, objective: str, llm_response: str) -> Dict[str, Any]:
        plan = self._try_parse_json(llm_response)
        if plan:
            plan["fallback_used"] = False
        else:
            plan = self._detect_tool_from_text(objective, llm_response)
            plan["fallback_used"] = True

        self.last_plan = plan
        self.plan_history.append({
            "objective": objective,
            "plan": plan
        })
        if len(self.plan_history) > 50:
            self.plan_history = self.plan_history[-50:]

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
        # PRIORITÉ 0-A : Ouverture d'application
        # ═══════════════════════════════════════

        app_result = self._detect_app_launch(objective_lower)
        if app_result:
            return app_result

        # ═══════════════════════════════════════
        # PRIORITÉ 0-B : Trading Engine
        # ═══════════════════════════════════════

        trading_result = self._detect_trading(objective_lower)
        if trading_result:
            return trading_result

        # ═══════════════════════════════════════
        # PRIORITÉ 1 : Détection directe
        # ═══════════════════════════════════════

        if re.search(r'\b(supprime|supprimer|efface|effacer|delete|remove|rm\s|del\s)\b', objective_lower):
            return {
                "tool": "conversation",
                "params": {"response": "🚫 Je ne suis pas autorisée à supprimer des fichiers."},
                "reasoning": "Demande de suppression → refus"
            }

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

        if re.search(r'(ping\s+\S|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower):
            cmd_match = re.search(r'(ping\s+[\w\.\-]+|ipconfig|systeminfo|tasklist|hostname|whoami)', objective_lower)
            command = cmd_match.group(0) if cmd_match else "echo commande non détectée"
            return {
                "tool": "shell_command",
                "params": {"command": command},
                "reasoning": "Commande shell détectée"
            }

        if re.search(r'connect.*mt5|mt5.*connect|connecte.*mt5|connecte-toi.*mt5|metatrader.*connect', objective_lower):
            return {
                "tool": "mt5_tool",
                "params": {"action": "connect"},
                "reasoning": "Connexion MT5 demandée"
            }

        if re.search(r'd[eé]compos|[eé]tape\s*par\s*[eé]tape|planifi.*action|fais.*plan|comment\s+proc[eé]der', objective_lower):
            return {
                "tool": "task_planner",
                "params": {"problem": objective},
                "reasoning": "Décomposition demandée"
            }

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
    # DÉTECTION APP LAUNCHER
    # ═══════════════════════════════════════

    def _detect_app_launch(self, objective_lower: str) -> Optional[Dict]:
        """Détecte si l'objectif est une demande d'ouverture d'application."""
        open_match = re.search(
            r'\b(ouvr[ei]|lance|d[eé]marre|start|open|démarre|demarre)\b'
            r'[\s\-]*'
            r'(?:moi\s+)?'
            r'(?:le\s+|la\s+|l\'|les\s+)?',
            objective_lower
        )

        if not open_match:
            return None

        after_verb = objective_lower[open_match.end():].strip()

        trading_keywords = [
            "trading", "backtest", "stratégie", "strategie", "strat",
            "forex", "bourse", "mt5", "metatrader", "optimis",
            "recherch", "session"
        ]
        if any(kw in after_verb for kw in trading_keywords):
            return None

        file_keywords = ["fichier", "file", "dossier", "répertoire", "repertoire"]
        if any(kw in after_verb for kw in file_keywords):
            return None

        for app in self.KNOWN_APPS:
            if app in after_verb:
                return {
                    "tool": "app_launcher",
                    "params": {"app_names": [app]},
                    "reasoning": f"Ouverture application: {app}"
                }

        app_name = after_verb.strip()
        app_name = re.sub(r'[.!?,;]+$', '', app_name).strip()

        if app_name and len(app_name) < 50 and not re.search(r'\b(et|puis|ensuite)\b', app_name):
            return {
                "tool": "app_launcher",
                "params": {"app_names": [app_name]},
                "reasoning": f"Ouverture application: {app_name}"
            }

        return None

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

        if re.search(
            r'cherch.*strat|trouv.*strat|optimis.*strat'
            r'|recherch.*strat|meilleur.*strat|search.*strat'
            r'|optimis.*trading|cherch.*trading|lance.*recherch.*trad'
            r'|lance.*optimis|trouv.*trading|trouve.*trad'
            r'|fait.*strat.*trad|fais.*strat.*trad'
            r'|fait.*trading.*rentable|fais.*trading.*rentable'
            r'|strat[eé]gi.*rentable'
            r'|fait.*strat[eé]gi.*rentable|fais.*strat[eé]gi.*rentable',
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

        if re.search(r'backtest|back.test', objective_lower):
            params = self._extract_trading_search_params(objective_lower)
            params["count"] = params.pop("population_size", 5)
            return {
                "tool": "trading_quick_test",
                "params": params,
                "reasoning": "Backtest → test rapide"
            }

        if has_symbol:
            params = self._extract_trading_search_params(objective_lower)
            params.setdefault("population_size", 10)
            params.setdefault("max_generations", 3)
            return {
                "tool": "trading_search",
                "params": params,
                "reasoning": "Trading avec symbole détecté"
            }

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

        all_symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
            "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "EURCAD", "EURCHF",
            "GBPCAD", "GBPCHF", "CADJPY", "CHFJPY", "CADCHF", "AUDCAD",
            "AUDCHF", "AUDJPY", "AUDNZD", "EURAUD", "EURNZD", "GBPAUD",
            "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "XAUUSD", "XAGUSD"
        ]

        multi_match = re.search(r'(\d+)\s*paire', text)
        if multi_match:
            num_pairs = int(multi_match.group(1))
            params["symbols"] = all_symbols[:min(num_pairs, len(all_symbols))]
        else:
            found = []
            for symbol in all_symbols:
                if symbol.lower() in text:
                    found.append(symbol)
            if found:
                params["symbols"] = found

        if not params.get("symbols") and re.search(r'multi[\s\-]*paire', text):
            params["symbols"] = all_symbols[:28]

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

        elif tool == "app_launcher":
            app_match = re.search(
                r'\b(ouvr[ei]|lance|d[eé]marre|start|open)\b'
                r'[\s\-]*(?:moi\s+)?(?:le\s+|la\s+|l\')?(.+)',
                objective, re.IGNORECASE
            )
            if app_match:
                app_name = app_match.group(2).strip().rstrip('.!?,;')
                return {"app_names": [app_name]}
            return {"app_names": [objective]}

        return {}


planner = Planner()