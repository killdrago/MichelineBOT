# micheline/tools/registry.py
"""
Registre centralisé de tous les tools disponibles.
"""

import logging
import importlib
import inspect
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger("micheline.tools.registry")


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    @property
    def tools(self):
        self._ensure_initialized()
        return self._tools

    @property
    def tool_names(self):
        self._ensure_initialized()
        return list(self._tools.keys())

    def register(self, name: str, function: Callable,
                 description: str = "", category: str = "general",
                 param_schema: dict = None, **kwargs):
        """
        Enregistre un tool.
        
        Args:
            name: nom unique
            function: fonction à appeler
            description: description pour le LLM
            category: catégorie
            param_schema: schéma des paramètres (utilisé par system_tools etc.)
            **kwargs: tout autre argument ignoré silencieusement
        """
        self._tools[name] = {
            "function": function,
            "description": description,
            "category": category,
        }
        if param_schema:
            self._tools[name]["param_schema"] = param_schema
        logger.debug(f"Tool enregistré : {name} [{category}]")

    def execute(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_initialized()
        if name not in self._tools:
            return {"error": f"Tool '{name}' non trouvé", "available": list(self._tools.keys())}
        params = params or {}
        try:
            result = self._tools[name]["function"](**params)
            logger.info(f"Tool '{name}' exécuté avec succès")
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Erreur tool '{name}': {e}")
            return {"error": str(e), "tool": name}

    def list_tools(self, category: Optional[str] = None) -> Dict[str, str]:
        self._ensure_initialized()
        tools = {}
        for name, info in self._tools.items():
            if category is None or info["category"] == category:
                tools[name] = info["description"]
        return tools

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        self._ensure_initialized()
        if name in self._tools:
            return {"name": name, "description": self._tools[name]["description"],
                    "category": self._tools[name]["category"]}
        return None

    def get_tool(self, name: str) -> Optional[Callable]:
        self._ensure_initialized()
        if name in self._tools:
            return self._tools[name]["function"]
        return None

    def has_tool(self, name: str) -> bool:
        self._ensure_initialized()
        return name in self._tools

    def get_tools_for_prompt(self) -> str:
        self._ensure_initialized()
        if not self._tools:
            return "Aucun outil disponible."
        lines = []
        categories = {}
        for name, info in self._tools.items():
            cat = info.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, info))
        for cat, tools_list in sorted(categories.items()):
            lines.append(f"\n📁 {cat.upper()} :")
            for name, info in tools_list:
                desc = info.get("description", "Pas de description")
                lines.append(f"  • {name} : {desc}")
        return "\n".join(lines)

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialized = True
            _do_register_all(self)

    def __contains__(self, name):
        self._ensure_initialized()
        return name in self._tools

    def __len__(self):
        self._ensure_initialized()
        return len(self._tools)

    def __getitem__(self, name):
        self._ensure_initialized()
        return self._tools[name]

    def __iter__(self):
        self._ensure_initialized()
        return iter(self._tools)

    def __repr__(self):
        return f"ToolRegistry({len(self._tools)} tools)"


# ──────────────────────────────────────────────
# Instance globale
# ──────────────────────────────────────────────

tool_registry = ToolRegistry()


# ──────────────────────────────────────────────
# Fonctions raccourcis
# ──────────────────────────────────────────────

def register_tool(name, function, description="", category="general", **kwargs):
    tool_registry.register(name, function, description, category, **kwargs)

def execute_tool(name, params=None):
    return tool_registry.execute(name, params)

def list_tools(category=None):
    return tool_registry.list_tools(category)

def get_tool_info(name):
    return tool_registry.get_tool_info(name)

def register_all_tools():
    tool_registry._initialized = False
    tool_registry._ensure_initialized()


# ──────────────────────────────────────────────
# Import sécurisé
# ──────────────────────────────────────────────

def _safe_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logger.debug(f"Module {module_path} non disponible: {e}")
        return None


# ──────────────────────────────────────────────
# ENREGISTREMENT COMPLET
# ──────────────────────────────────────────────

def _do_register_all(registry: ToolRegistry):
    logger.info("═══ Enregistrement des tools ═══")
    total = 0

    # ── file_tools : register_file_tools(registry) ──
    try:
        from micheline.tools.file_tools import register_file_tools
        before = len(registry._tools)
        register_file_tools(registry)
        added = len(registry._tools) - before
        total += added
        logger.info(f"  ✅ file_tools : {added} tools")
    except ImportError as e:
        logger.warning(f"  ⚠️ file_tools import: {e}")
        # Fallback : auto-détection
        mod = _safe_import("micheline.tools.file_tools")
        if mod:
            added = _auto_register_module(registry, mod, "file")
            total += added
    except Exception as e:
        logger.warning(f"  ⚠️ file_tools erreur: {e}")
        mod = _safe_import("micheline.tools.file_tools")
        if mod:
            added = _auto_register_module(registry, mod, "file")
            total += added

    # ── system_tools : register_system_tools(registry) ──
    try:
        from micheline.tools.system_tools import register_system_tools
        before = len(registry._tools)
        register_system_tools(registry)
        added = len(registry._tools) - before
        total += added
        logger.info(f"  ✅ system_tools : {added} tools")
    except ImportError as e:
        logger.warning(f"  ⚠️ system_tools import: {e}")
        mod = _safe_import("micheline.tools.system_tools")
        if mod:
            added = _auto_register_module(registry, mod, "system")
            total += added
    except Exception as e:
        logger.warning(f"  ⚠️ system_tools erreur: {e}")
        mod = _safe_import("micheline.tools.system_tools")
        if mod:
            added = _auto_register_module(registry, mod, "system")
            total += added

    # ── mt5_tool ──
    try:
        mt5_mod = _safe_import("micheline.tools.mt5_tool")
        if mt5_mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mt5_mod):
                if fname.startswith("register") and callable(getattr(mt5_mod, fname)):
                    getattr(mt5_mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mt5_mod, "mt5")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ mt5_tool : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ mt5_tool: {e}")

    # ── shell_tool ──
    try:
        mod = _safe_import("micheline.tools.shell_tool")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "shell")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ shell_tool : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ shell_tool: {e}")

    # ── code_executor ──
    try:
        mod = _safe_import("micheline.tools.code_executor")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "code")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ code_executor : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ code_executor: {e}")

    # ── web_search_tool ──
    try:
        mod = _safe_import("micheline.tools.web_search_tool")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "web")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ web_search_tool : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ web_search_tool: {e}")

    # ── app_launcher ──
    try:
        mod = _safe_import("micheline.tools.app_launcher")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "app")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ app_launcher : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ app_launcher: {e}")

    # ── task_planner_tool ──
    try:
        mod = _safe_import("micheline.tools.task_planner_tool")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "planner")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ task_planner_tool : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ task_planner_tool: {e}")

    # ── memory ──
    try:
        mod = _safe_import("micheline.memory.memory")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "memory")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ memory : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ memory: {e}")

    # ── security/path_guard ──
    try:
        mod = _safe_import("micheline.security.path_guard")
        if mod:
            before = len(registry._tools)
            found = False
            for fname in dir(mod):
                if fname.startswith("register") and callable(getattr(mod, fname)):
                    getattr(mod, fname)(registry)
                    found = True
                    break
            if not found:
                _auto_register_module(registry, mod, "security")
            added = len(registry._tools) - before
            total += added
            if added > 0:
                logger.info(f"  ✅ security : {added} tools")
    except Exception as e:
        logger.warning(f"  ⚠️ security: {e}")

    # ── Trading Engine (Phase 6) ──
    trading_count = _register_trading_engine(registry)
    total += trading_count

    logger.info(f"═══ Total : {total} tools enregistrés ═══")


def _auto_register_module(registry: ToolRegistry, module, category: str) -> int:
    count = 0
    module_name = module.__name__
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not callable(obj):
            continue
        if inspect.isclass(obj):
            continue
        if inspect.ismodule(obj):
            continue
        if name.startswith("register"):
            continue
        if hasattr(obj, "__module__") and obj.__module__ != module_name:
            continue
        doc = inspect.getdoc(obj) or f"Fonction {name}"
        description = doc.split("\n")[0].strip()
        registry.register(name, obj, description, category)
        count += 1
    if count > 0:
        logger.info(f"  ✅ {module_name} (auto) : {count} tools")
    return count


def _register_trading_engine(registry: ToolRegistry) -> int:
    count = 0

    # Chercher backtest dans mt5_tool
    backtest_fn = None
    mt5_mod = _safe_import("micheline.tools.mt5_tool")
    if mt5_mod:
        for fname in ["run_backtest", "backtest", "tool_backtest"]:
            if hasattr(mt5_mod, fname):
                backtest_fn = getattr(mt5_mod, fname)
                break

    if backtest_fn is None:
        import random
        import hashlib

        def simulated_backtest(config=None, **kwargs):
            """Backtest simulé."""
            config = config or kwargs
            seed_str = str(sorted(str(config).encode()))
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            trades = rng.randint(20, 150)
            winrate = round(rng.uniform(30, 70), 2)
            profit = round(rng.uniform(-2000, 5000), 2)
            drawdown = round(rng.uniform(5, 40), 2)
            return {
                "profit": profit, "drawdown": drawdown,
                "trades": trades, "winrate": winrate,
                "gross_profit": round(max(0, profit * 1.5), 2),
                "gross_loss": round(-abs(min(0, profit * 0.5)), 2),
                "initial_deposit": 10000.0, "simulated": True,
            }
        backtest_fn = simulated_backtest

    try:
        # ← CORRECTION ICI : micheline.trading, pas trading
        from micheline.trading.engine import TradingEngine

        store_fn = None
        retrieve_fn = None
        mem_mod = _safe_import("micheline.memory.memory")
        if mem_mod:
            for cname in dir(mem_mod):
                obj = getattr(mem_mod, cname)
                if inspect.isclass(obj) and "memory" in cname.lower():
                    try:
                        mem_instance = obj()
                        if hasattr(mem_instance, "store_experience"):
                            store_fn = mem_instance.store_experience
                        if hasattr(mem_instance, "search_experiences"):
                            retrieve_fn = mem_instance.search_experiences
                    except Exception:
                        pass
                    break

        engine = TradingEngine(
            run_backtest_fn=backtest_fn,
            store_fn=store_fn,
            retrieve_fn=retrieve_fn,
        )

        registry.register("trading_search", engine.search_strategy,
                          "Rechercher une stratégie de trading optimale", "trading")
        registry.register("trading_quick_test", engine.quick_test,
                          "Test rapide de stratégies aléatoires", "trading")
        registry.register("trading_evaluate", engine.evaluate_strategy,
                          "Évaluer une stratégie de trading", "trading")
        registry.register("trading_improve", engine.improve_strategy,
                          "Améliorer une stratégie par mutations", "trading")
        registry.register("trading_top_strategies", engine.get_top_strategies,
                          "Meilleures stratégies trouvées", "trading")
        registry.register("trading_report", engine.get_session_report,
                          "Rapport de session trading", "trading")
        count = 6
        logger.info(f"  ✅ Trading Engine : {count} tools")

    except ImportError as e:
        logger.warning(f"  ⚠️ Trading engine: {e}")
    except Exception as e:
        logger.warning(f"  ⚠️ Trading engine erreur: {e}")

    return count