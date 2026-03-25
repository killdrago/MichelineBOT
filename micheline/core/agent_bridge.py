"""
micheline/core/agent_bridge.py

MichelineBridge avec support MULTI-ACTIONS :
"fait moi une stratégie EURUSD et ouvre moi paint"
→ Action 1: trading_search (EURUSD)
→ Action 2: app_launcher (paint)
"""

import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger("micheline.agent_bridge")


class AgentExecutor:
    def __init__(self, registry=None):
        self.registry = registry
        self.llm_client = None

    def execute(self, tool_name: str, params: dict) -> Dict[str, Any]:
        if self.registry is None:
            return {"success": False, "error": "Registry non initialisé"}
        return self.registry.execute(tool_name, params)


class AgentEvaluator:
    def __init__(self):
        self.llm_client = None

    def evaluate(self, result: dict, tool_name: str) -> tuple:
        if not result or not isinstance(result, dict):
            return False, "Résultat invalide"
        if not result.get("success", False):
            return False, result.get("error", "Échec sans détail")
        if tool_name in ("trading_search", "trading_generate"):
            score = result.get("best_score", 0)
            if score >= 15:
                return True, f"Score acceptable: {score:.1f}"
            else:
                return False, f"Score insuffisant: {score:.1f}"
        return True, "Résultat obtenu"


class AgentCore:
    def __init__(self, planner, executor, evaluator):
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator


class MichelineBridge:

    def __init__(self, llm_client=None, log_callback=None, agent_mode=True, max_iterations=5):
        self.llm_client = llm_client
        self.log_callback = log_callback or (lambda msg: None)
        self.max_iterations = max_iterations
        self.agent_mode = agent_mode

        try:
            from micheline.core.planner import Planner
        except ImportError:
            from core.planner import Planner

        try:
            from micheline.tools.registry import ToolRegistry
        except ImportError:
            from tools.registry import ToolRegistry

        self.tool_registry = ToolRegistry()
        planner = Planner()
        executor = AgentExecutor(registry=self.tool_registry)
        evaluator = AgentEvaluator()

        self.agent = AgentCore(planner=planner, executor=executor, evaluator=evaluator)
        self.conversation_history = []
        self._initialized = False
        self._log("MichelineBridge créé")

    def _log(self, msg: str):
        try:
            self.log_callback(msg)
        except Exception:
            pass
        logger.info(msg)

    def initialize(self):
        if self._initialized:
            return
        try:
            self.tool_registry.register_all()
            tools = self.tool_registry.list_tools()
            self.agent.planner.update_tools(tools)
            self._initialized = True
            self._log(f"Bridge initialisé avec {len(tools)} outils")
        except Exception as e:
            logger.error(f"Erreur init bridge: {e}", exc_info=True)
            self._initialized = True

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Traite une entrée utilisateur.
        SUPPORTE LES MULTI-ACTIONS via split_objectives().
        """
        if not self._initialized:
            self.initialize()

        self._log(f"📨 Traitement : {user_input[:80]}...")
        self._log(f"🎯 Objectif : {user_input}")

        # ═══════════════════════════════════
        # SPLIT MULTI-ACTIONS
        # ═══════════════════════════════════
        sub_objectives = self.agent.planner.split_objectives(user_input)

        if len(sub_objectives) > 1:
            self._log(f"🔀 Multi-actions détectées: {len(sub_objectives)} sous-objectifs")

        all_responses = []
        all_logs = []
        total_iterations = 0
        overall_status = "success"

        for obj_idx, sub_objective in enumerate(sub_objectives):
            if len(sub_objectives) > 1:
                self._log(f"═══ Sous-objectif {obj_idx + 1}/{len(sub_objectives)}: {sub_objective[:60]} ═══")

            result = self._process_single_objective(sub_objective)

            total_iterations += result.get("iterations", 0)
            all_logs.extend(result.get("logs", []))

            if result.get("status") == "failed":
                overall_status = "partial"

            response_text = result.get("response", "")
            if response_text:
                all_responses.append(response_text)

        # Combiner les réponses
        if len(all_responses) == 1:
            combined_response = all_responses[0]
        elif len(all_responses) > 1:
            combined_response = "\n\n".join(
                f"{'─' * 40}\n{resp}" for resp in all_responses
            )
        else:
            combined_response = "Aucun résultat."

        return {
            "status": overall_status,
            "response": combined_response,
            "iterations": total_iterations,
            "logs": all_logs
        }

    def _process_single_objective(self, objective: str) -> Dict[str, Any]:
        """
        Traite UN SEUL objectif via la boucle agent.
        """
        self.agent.planner.reset()

        last_error = None
        best_result = None
        best_score = -1
        logs = []
        iterations = 0

        for i in range(self.max_iterations):
            iterations = i + 1
            self._log(f"━━━ Itération {iterations}/{self.max_iterations} ━━━")

            # PLANIFICATION
            self._log("📋 Planification...")
            try:
                plan = self.agent.planner.plan(
                    objective=objective,
                    context={"iteration": i, "last_error": last_error},
                    llm=self.llm_client
                )
            except Exception as e:
                plan = {"tool": "none", "params": {}, "reasoning": str(e), "fallback": None}

            tool_name = plan.get("tool", "none")
            params = plan.get("params", {})
            reasoning = plan.get("reasoning", "")
            fallback = plan.get("fallback")

            self._log(f"   Outil : {tool_name}")
            self._log(f"   Params : {params}")
            self._log(f"   Raisonnement : {reasoning}")

            # CONVERSATION (réponse directe du planner)
            if tool_name == "conversation":
                response = params.get("response", reasoning)
                logs.append(f"Conversation directe: {reasoning}")
                return {
                    "status": "success",
                    "response": response,
                    "iterations": iterations,
                    "logs": logs
                }

            # LLM DIRECT
            if tool_name == "llm_direct":
                response = self._handle_llm_direct(params.get("prompt", objective))
                if response:
                    logs.append(f"LLM direct (itération {iterations})")
                    return {"status": "success", "response": response, "iterations": iterations, "logs": logs}
                last_error = "LLM direct échoué"
                continue

            # AUCUN OUTIL
            if tool_name == "none":
                last_error = "Aucun outil trouvé"
                continue

            # EXÉCUTION
            self._log("⚡ Exécution...")
            result = self.agent.executor.execute(tool_name, params)

            success = result and isinstance(result, dict) and result.get("success", False)
            self._log(f"   Succès : {success}")

            if success:
                self.agent.planner.record_success(tool_name)

                if "best_score" in result:
                    self._log(f"   Score: {result.get('best_score', '?')}")

                # ÉVALUATION
                self._log("🔍 Évaluation...")
                is_complete, eval_msg = self.agent.evaluator.evaluate(result, tool_name)
                self._log(f"   Complet : {is_complete}")
                self._log(f"   Évaluation : {eval_msg}")

                if is_complete:
                    response = self._format_response(result, tool_name)
                    logs.append(f"Succès avec {tool_name} (itération {iterations})")
                    return {"status": "success", "response": response, "iterations": iterations, "logs": logs}

                current_score = result.get("best_score", 0)
                if current_score > best_score:
                    best_score = current_score
                    best_result = result

                last_error = eval_msg

            else:
                error_msg = result.get("error", "Erreur inconnue") if isinstance(result, dict) else "Invalide"
                self._log(f"   Résultat : {error_msg}")
                self.agent.planner.record_failure(tool_name, params, error_msg)
                last_error = error_msg
                logs.append(f"Échec {tool_name}: {error_msg}")

                # Fallback
                if fallback and fallback != tool_name:
                    self._log(f"   🔄 Fallback: {fallback}")
                    fb_result = self.agent.executor.execute(fallback, params)
                    if fb_result and isinstance(fb_result, dict) and fb_result.get("success"):
                        self.agent.planner.record_success(fallback)
                        response = self._format_response(fb_result, fallback)
                        logs.append(f"Fallback {fallback} réussi")
                        return {"status": "success", "response": response, "iterations": iterations, "logs": logs}

                self._log(f"   Évaluation : Échec de {tool_name}, réessai possible")

        # ÉPUISÉ
        if best_result:
            response = self._format_response(best_result, "partial")
            return {"status": "partial", "response": response, "iterations": iterations, "logs": logs}

        llm_response = self._handle_llm_direct(
            f"L'utilisateur demande: {objective}\nÉchec après {iterations} tentatives.\nDernière erreur: {last_error}"
        )
        if llm_response:
            return {"status": "llm_fallback", "response": llm_response, "iterations": iterations, "logs": logs}

        return {
            "status": "failed",
            "response": f"⚠️ Échec après {iterations} tentatives.\nDernière erreur: {last_error}",
            "iterations": iterations,
            "logs": logs
        }

    def _handle_llm_direct(self, prompt: str) -> Optional[str]:
        if not self.llm_client:
            return None
        try:
            if hasattr(self.llm_client, 'generate'):
                response = self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, 'chat'):
                response, _, _ = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="Tu es Micheline, une IA locale.",
                    temperature=0.3, max_tokens=900
                )
            else:
                return None
            return response.strip() if response and isinstance(response, str) else None
        except Exception as e:
            logger.error(f"LLM direct échoué: {e}")
            return None

    def _format_response(self, result: dict, tool_name: str) -> str:
        if "formatted" in result and result["formatted"]:
            return result["formatted"]

        # Réponse directe (app_launcher, shell, etc.)
        if "output" in result:
            return str(result["output"])
        if "message" in result:
            return str(result["message"])

        if tool_name in ("trading_search", "trading_generate", "partial"):
            try:
                from micheline.tools.trading_tools import format_strategy_summary
                return format_strategy_summary(result)
            except ImportError:
                try:
                    from tools.trading_tools import format_strategy_summary
                    return format_strategy_summary(result)
                except ImportError:
                    pass

        lines = [f"📋 Résultat ({tool_name}):"]
        for key, value in result.items():
            if key not in ("success", "execution_time", "formatted", "trade_results", "equity_curve"):
                if isinstance(value, (list, dict)):
                    lines.append(f"  • {key}: [{type(value).__name__}]")
                else:
                    lines.append(f"  • {key}: {value}")
        return "\n".join(lines)

    def get_available_tools(self) -> List[str]:
        return self.tool_registry.list_tools()

    def get_tools_description(self) -> str:
        return self.tool_registry.get_tools_description()