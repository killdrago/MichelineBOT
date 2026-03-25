"""
micheline/tools/registry.py

Registre central de tous les outils disponibles pour l'agent Micheline.
Tous les appels d'outils passent par ce registre.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger("micheline.tools.registry")


class ToolDefinition:
    """Définition d'un outil enregistré."""

    def __init__(self, name, func, description="", parameters=None, category="general"):
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters or {}
        self.category = category
        self.call_count = 0
        self.total_time = 0.0
        self.last_error = None

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "call_count": self.call_count,
            "avg_time": round(self.total_time / max(1, self.call_count), 3)
        }


class ToolRegistry:
    """Registre central des outils."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
        self._memory_store: list = []
        logger.info("ToolRegistry initialisé")

    # ═══════════════════════════════════
    # MÉTHODES DE BASE
    # ═══════════════════════════════════

    def register(self, name, func, description="", parameters=None, category="general"):
        tool = ToolDefinition(name, func, description, parameters, category)
        self._tools[name] = tool
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        logger.debug(f"Outil enregistré: {name} [{category}]")

    def get(self, name):
        tool = self._tools.get(name)
        return tool.func if tool else None

    def execute(self, name, params=None):
        params = params or {}
        tool = self._tools.get(name)
        if tool is None:
            logger.error(f"Outil inconnu: {name}")
            return {
                "success": False,
                "error": f"Outil '{name}' non trouvé",
                "available_tools": list(self._tools.keys())
            }

        start_time = time.time()
        try:
            result = tool.func(params)
            elapsed = time.time() - start_time
            tool.call_count += 1
            tool.total_time += elapsed
            tool.last_error = None

            if result is None:
                return {"success": False, "error": f"'{name}' a retourné None", "execution_time": elapsed}
            if not isinstance(result, dict):
                return {"success": True, "data": result, "execution_time": elapsed}
            result["execution_time"] = elapsed
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            tool.call_count += 1
            tool.total_time += elapsed
            tool.last_error = str(e)
            logger.error(f"Erreur {name}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "tool": name, "execution_time": elapsed}

    def list_tools(self):
        return list(self._tools.keys())

    def list_by_category(self, category):
        return self._categories.get(category, [])

    def get_all_categories(self):
        return dict(self._categories)

    def get_tools_description(self):
        lines = ["Outils disponibles:\n"]
        for category, tool_names in sorted(self._categories.items()):
            lines.append(f"\n[{category.upper()}]")
            for name in tool_names:
                tool = self._tools[name]
                params_str = ", ".join(
                    f"{k}: {v.get('type', 'any')}"
                    for k, v in tool.parameters.items()
                ) if tool.parameters else "aucun"
                lines.append(f"  • {name}: {tool.description}")
                lines.append(f"    Paramètres: {params_str}")
        return "\n".join(lines)

    def get_stats(self):
        return {
            "total_tools": len(self._tools),
            "categories": list(self._categories.keys()),
            "tools": {name: tool.to_dict() for name, tool in self._tools.items()}
        }

    # ═══════════════════════════════════
    # ENREGISTREMENT DE TOUS LES OUTILS
    # ═══════════════════════════════════

    def register_all(self):
        logger.info("Enregistrement de tous les outils...")
        self._register_trading_tools()
        self._register_file_tools()
        self._register_system_tools()
        self._register_web_tools()
        self._register_memory_tools()
        self._register_analysis_tools()
        logger.info(f"Total: {len(self._tools)} outils dans {len(self._categories)} catégories")

    def _safe_call(self, func, params):
        try:
            result = func(params)
            if result is None:
                return {"success": False, "error": "Fonction a retourné None"}
            if not isinstance(result, dict):
                return {"success": True, "data": result}
            return result
        except Exception as e:
            logger.error(f"Erreur safe_call: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ── TRADING ──

    def _register_trading_tools(self):
        try:
            try:
                from micheline.tools.trading_tools import (
                    trading_search, trading_generate, run_backtest,
                    generate_random_strategy, format_strategy_summary
                )
            except ImportError:
                from tools.trading_tools import (
                    trading_search, trading_generate, run_backtest,
                    generate_random_strategy, format_strategy_summary
                )

            self.register(
                name="trading_search",
                func=lambda p: self._safe_call(trading_search, p),
                description="Recherche de stratégies de trading optimisées par algorithme génétique",
                parameters={
                    "symbols": {"type": "list", "description": "Symboles forex", "required": True},
                    "population_size": {"type": "integer", "required": False, "default": 10},
                    "max_generations": {"type": "integer", "required": False, "default": 3}
                },
                category="trading"
            )

            self.register(
                name="trading_generate",
                func=lambda p: self._safe_call(trading_generate, p),
                description="Génère UNE stratégie simple (fallback rapide)",
                parameters={
                    "symbol": {"type": "string", "required": False, "default": "EURUSD"}
                },
                category="trading"
            )

            self.register(
                name="trading_backtest",
                func=lambda p: self._safe_call(
                    lambda params: {"success": True, **run_backtest(params.get("strategy", {}))},
                    p
                ),
                description="Lance un backtest sur une stratégie",
                parameters={"strategy": {"type": "dict", "required": True}},
                category="trading"
            )

            self.register(
                name="trading_generate_random",
                func=lambda p: self._safe_call(
                    lambda params: {"success": True, "strategy": generate_random_strategy(params.get("symbol", "EURUSD"))},
                    p
                ),
                description="Génère une config stratégie aléatoire sans backtest",
                parameters={"symbol": {"type": "string", "required": False, "default": "EURUSD"}},
                category="trading"
            )

            logger.info("✅ Outils trading enregistrés")

        except ImportError as e:
            logger.error(f"❌ Import trading tools échoué: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement trading: {e}", exc_info=True)

    # ── FICHIERS ──

    def _register_file_tools(self):
        try:
            has_file_tools = False
            try:
                try:
                    from micheline.tools.file_tools import read_file, write_file, list_directory
                except ImportError:
                    from tools.file_tools import read_file, write_file, list_directory
                has_file_tools = True
            except ImportError:
                pass

            if has_file_tools:
                self.register("file_read", lambda p: self._safe_call(read_file, p),
                              "Lit un fichier", {"path": {"type": "string", "required": True}}, "files")
                self.register("file_write", lambda p: self._safe_call(write_file, p),
                              "Écrit dans un fichier",
                              {"path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}},
                              "files")
                self.register("file_list", lambda p: self._safe_call(list_directory, p),
                              "Liste un répertoire", {"path": {"type": "string", "required": True}}, "files")
            else:
                self.register("file_read", lambda p: self._file_read_fallback(p),
                              "Lit un fichier", {"path": {"type": "string", "required": True}}, "files")
                self.register("file_write", lambda p: self._file_write_fallback(p),
                              "Écrit dans un fichier",
                              {"path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}},
                              "files")

            logger.info("✅ Outils fichiers enregistrés")
        except Exception as e:
            logger.error(f"❌ Erreur fichiers: {e}")

    # ── SYSTÈME ──

    def _register_system_tools(self):
        try:
            self.register("system_info", lambda p: self._safe_call(self._get_system_info, p),
                          "Informations système", {}, "system")
            self.register("system_time", lambda p: {"success": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": time.time()},
                          "Heure actuelle", {}, "system")
            self.register("tool_list", lambda p: {"success": True, "tools": self.list_tools(), "categories": self.get_all_categories(), "total": len(self._tools)},
                          "Liste des outils", {}, "system")
            self.register("tool_stats", lambda p: {"success": True, **self.get_stats()},
                          "Stats des outils", {}, "system")
            logger.info("✅ Outils système enregistrés")
        except Exception as e:
            logger.error(f"❌ Erreur système: {e}")

    # ── WEB ──

    def _register_web_tools(self):
        try:
            self.register("web_search", lambda p: self._safe_call(self._web_search, p),
                          "Recherche web", {"query": {"type": "string", "required": True}}, "web")
            self.register("web_fetch", lambda p: self._safe_call(self._web_fetch, p),
                          "Récupère une URL", {"url": {"type": "string", "required": True}}, "web")
            logger.info("✅ Outils web enregistrés")
        except Exception as e:
            logger.error(f"❌ Erreur web: {e}")

    # ── MÉMOIRE ──

    def _register_memory_tools(self):
        try:
            has_memory = False
            try:
                try:
                    from micheline.memory.memory import store, retrieve
                except ImportError:
                    from memory.memory import store, retrieve
                has_memory = True
            except ImportError:
                pass

            if has_memory:
                self.register("memory_store", lambda p: self._safe_call(lambda x: {"success": True, "stored": store(x)}, p),
                              "Stocke en mémoire", {"data": {"type": "dict", "required": True}}, "memory")
                self.register("memory_retrieve", lambda p: self._safe_call(lambda x: {"success": True, "results": retrieve(x.get("query", ""))}, p),
                              "Recherche en mémoire", {"query": {"type": "string", "required": True}}, "memory")
            else:
                self.register("memory_store", lambda p: self._memory_store_fallback(p),
                              "Stocke en mémoire (RAM)", {"data": {"type": "dict", "required": True}}, "memory")
                self.register("memory_retrieve", lambda p: self._memory_retrieve_fallback(p),
                              "Recherche en mémoire (RAM)", {"query": {"type": "string", "required": True}}, "memory")

            logger.info("✅ Outils mémoire enregistrés")
        except Exception as e:
            logger.error(f"❌ Erreur mémoire: {e}")

    # ── ANALYSE ──

    def _register_analysis_tools(self):
        try:
            self.register("analyze_strategy", lambda p: self._safe_call(self._analyze_strategy, p),
                          "Analyse une stratégie", {"result": {"type": "dict", "required": True}}, "analysis")
            self.register("compare_strategies", lambda p: self._safe_call(self._compare_strategies, p),
                          "Compare des stratégies", {"strategies": {"type": "list", "required": True}}, "analysis")
            logger.info("✅ Outils analyse enregistrés")
        except Exception as e:
            logger.error(f"❌ Erreur analyse: {e}")

    # ═══════════════════════════════════
    # IMPLÉMENTATIONS INTERNES
    # ═══════════════════════════════════

    def _get_system_info(self, params):
        import platform
        info = {
            "success": True,
            "os": platform.system(),
            "os_version": platform.version(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        }
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram_total_mb"] = round(mem.total / 1024 / 1024)
            info["ram_used_pct"] = mem.percent
            info["ram_available_mb"] = round(mem.available / 1024 / 1024)
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_pct"] = psutil.cpu_percent(interval=0.5)
        except ImportError:
            info["note"] = "psutil non installé"
        return info

    def _web_search(self, params):
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "'query' requis"}
        try:
            import urllib.request, urllib.parse, json
            encoded = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Micheline/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            results = []
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", ""), "text": data["AbstractText"], "url": data.get("AbstractURL", "")})
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({"title": topic.get("Text", "")[:100], "text": topic.get("Text", ""), "url": topic.get("FirstURL", "")})
            return {"success": True, "query": query, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def _web_fetch(self, params):
        url = params.get("url", "")
        if not url:
            return {"success": False, "error": "'url' requis"}
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Micheline/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="replace")
            return {"success": True, "url": url, "content": content[:5000], "length": len(content), "truncated": len(content) > 5000}
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}

    def _file_read_fallback(self, params):
        path = params.get("path", "")
        if not path:
            return {"success": False, "error": "'path' requis"}
        try:
            try:
                from micheline.security.path_guard import is_allowed
            except ImportError:
                try:
                    from security.path_guard import is_allowed
                except ImportError:
                    is_allowed = lambda p: True
            if not is_allowed(path):
                return {"success": False, "error": f"Accès refusé: {path}"}
        except Exception:
            pass
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"success": True, "path": path, "content": content, "length": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    def _file_write_fallback(self, params):
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return {"success": False, "error": "'path' requis"}
        try:
            try:
                from micheline.security.path_guard import is_allowed
            except ImportError:
                try:
                    from security.path_guard import is_allowed
                except ImportError:
                    is_allowed = lambda p: True
            if not is_allowed(path):
                return {"success": False, "error": f"Accès refusé: {path}"}
        except Exception:
            pass
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": path, "written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    def _memory_store_fallback(self, params):
        data = params.get("data", params)
        self._memory_store.append({"timestamp": time.time(), "data": data})
        return {"success": True, "stored": True, "total_entries": len(self._memory_store)}

    def _memory_retrieve_fallback(self, params):
        query = str(params.get("query", "")).lower()
        results = [e for e in self._memory_store if query in str(e.get("data", "")).lower()]
        return {"success": True, "query": query, "results": results[-10:], "count": len(results)}

    def _analyze_strategy(self, params):
        result = params.get("result", {})
        if not result:
            return {"success": False, "error": "'result' requis"}
        try:
            try:
                from micheline.trading.metrics import evaluate_strategy
            except ImportError:
                from trading.metrics import evaluate_strategy
            score = evaluate_strategy(result)
        except ImportError:
            score = 0
        analysis = {
            "success": True,
            "score": score,
            "verdict": "excellent" if score >= 60 else "acceptable" if score >= 40 else "faible",
            "recommendations": []
        }
        if result.get("winrate", 0) < 0.4:
            analysis["recommendations"].append("Winrate faible")
        if result.get("drawdown", 0) > 500:
            analysis["recommendations"].append("Drawdown élevé")
        if result.get("trades", 0) < 30:
            analysis["recommendations"].append("Trop peu de trades")
        return analysis

    def _compare_strategies(self, params):
        strategies = params.get("strategies", [])
        if not strategies:
            return {"success": False, "error": "'strategies' requis"}
        try:
            try:
                from micheline.trading.metrics import evaluate_strategy
            except ImportError:
                from trading.metrics import evaluate_strategy
            scored = [{"index": i, "score": evaluate_strategy(s), **s} for i, s in enumerate(strategies)]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return {"success": True, "ranking": scored, "best_index": scored[0]["index"]}
        except Exception as e:
            return {"success": False, "error": str(e)}