"""
Agent Bridge — Pont entre main.py et le système agent.
Emplacement : micheline/core/agent_bridge.py
COMPATIBLE avec main.py (MichelineBridge)
Phase 6 — Support multi-actions
"""

import re
import json
import traceback
from typing import Dict, Any, Tuple, List, Optional

from micheline.tools.registry import tool_registry
from micheline.core.planner import planner as planner_instance
from micheline.core.executor import executor as executor_instance


class AgentLoop:
    """Boucle agent : Plan → Execute → Evaluate."""

    def __init__(self, llm_client=None, log_callback=None):
        self.llm_client = llm_client
        self.log_callback = log_callback or (lambda msg: None)
        self.planner = AgentPlanner(llm_client=llm_client)
        self.executor = AgentExecutor(llm_client=llm_client)
        self.evaluator = AgentEvaluator(llm_client=llm_client)
        self.max_iterations = 5

    def run(self, objective: str) -> Dict[str, Any]:
        logs = []
        all_results = []
        status = "error"

        # ── Détecter les sous-objectifs multiples ──
        sub_objectives = self._split_objectives(objective)

        if len(sub_objectives) > 1:
            self.log_callback(f"�� {len(sub_objectives)} actions détectées")
            logs.append(f"🎯 {len(sub_objectives)} actions détectées")

            for idx, sub_obj in enumerate(sub_objectives, 1):
                self.log_callback(f"━━━ Action {idx}/{len(sub_objectives)} : {sub_obj} ━━━")
                logs.append(f"━━━ Action {idx}/{len(sub_objectives)} : {sub_obj} ━━━")

                try:
                    plan = self._plan(sub_obj, all_results)
                    tool_name = plan.get("tool", "conversation")
                    params = plan.get("params", {})
                    reasoning = plan.get("reasoning", "")

                    self.log_callback(f"   Outil : {tool_name}")
                    self.log_callback(f"   Params : {json.dumps(params, ensure_ascii=False, default=str)[:200]}")
                    if reasoning:
                        self.log_callback(f"   Raisonnement : {reasoning[:150]}")

                    exec_result = executor_instance.execute(plan)
                    result_text = exec_result.get("result", "")
                    success = exec_result.get("success", False)
                    actual_tool = exec_result.get("tool_used", tool_name)

                    self.log_callback(f"   Succès : {success}")
                    self.log_callback(f"   Résultat : {str(result_text)[:300]}")

                    all_results.append({
                        "iteration": idx,
                        "tool": actual_tool,
                        "success": success,
                        "result": result_text,
                    })

                    if success:
                        status = "success"

                except Exception as e:
                    error_msg = f"❌ Erreur action {idx} : {type(e).__name__}: {e}"
                    self.log_callback(error_msg)
                    logs.append(error_msg)
                    all_results.append({
                        "iteration": idx,
                        "tool": "error",
                        "success": False,
                        "result": str(e),
                    })

            response = self._build_final_response(objective, all_results)
            return {
                "response": response,
                "status": status,
                "iterations": len(all_results),
                "logs": logs,
            }

        # ── Mode normal : une seule action avec boucle ──
        self.log_callback(f"🎯 Objectif : {objective}")
        logs.append(f"🎯 Objectif : {objective}")

        for i in range(self.max_iterations):
            self.log_callback(f"━━━ Itération {i + 1}/{self.max_iterations} ━━━")
            logs.append(f"━━━ Itération {i + 1}/{self.max_iterations} ━━━")

            try:
                self.log_callback("📋 Planification...")
                logs.append("📋 Planification...")

                plan = self._plan(objective, all_results)
                tool_name = plan.get("tool", "conversation")
                params = plan.get("params", {})
                reasoning = plan.get("reasoning", "")

                self.log_callback(f"   Outil : {tool_name}")
                self.log_callback(f"   Params : {json.dumps(params, ensure_ascii=False, default=str)[:200]}")
                if reasoning:
                    self.log_callback(f"   Raisonnement : {reasoning[:150]}")
                logs.append(f"   Outil : {tool_name} | Params : {json.dumps(params, ensure_ascii=False, default=str)[:200]}")

                self.log_callback("⚡ Exécution...")
                logs.append("⚡ Exécution...")

                exec_result = executor_instance.execute(plan)
                result_text = exec_result.get("result", "")
                success = exec_result.get("success", False)
                actual_tool = exec_result.get("tool_used", tool_name)

                self.log_callback(f"   Succès : {success}")
                self.log_callback(f"   Résultat : {str(result_text)[:300]}")
                logs.append(f"   Succès : {success} | Résultat : {str(result_text)[:300]}")

                all_results.append({
                    "iteration": i + 1,
                    "tool": actual_tool,
                    "success": success,
                    "result": result_text,
                })

                self.log_callback("🔍 Évaluation...")
                logs.append("🔍 Évaluation...")

                evaluation = self._evaluate(objective, all_results)
                is_complete = evaluation.get("complete", False)
                eval_reasoning = evaluation.get("reasoning", "")

                self.log_callback(f"   Complet : {is_complete}")
                if eval_reasoning:
                    self.log_callback(f"   Évaluation : {eval_reasoning[:150]}")
                logs.append(f"   Complet : {is_complete} | {eval_reasoning[:150]}")

                if is_complete or tool_name == "conversation":
                    status = "success"
                    break

            except Exception as e:
                error_msg = f"❌ Erreur itération {i + 1} : {type(e).__name__}: {e}"
                self.log_callback(error_msg)
                logs.append(error_msg)
                traceback.print_exc()
                all_results.append({
                    "iteration": i + 1,
                    "tool": "error",
                    "success": False,
                    "result": str(e),
                })
                continue

        if status != "success" and all_results:
            status = "partial"

        response = self._build_final_response(objective, all_results)

        return {
            "response": response,
            "status": status,
            "iterations": len(all_results),
            "logs": logs,
        }

    def _split_objectives(self, objective: str) -> List[str]:
        """
        Détecte si le message contient plusieurs actions distinctes.
        Utilise un lookahead pour ne pas consommer les mots-clés.
        """
        text = objective.strip()

        # Mots-clés qui indiquent une NOUVELLE action
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

        # Pattern : "et/puis/ensuite" suivi d'un mot-clé d'action
        # Le lookahead (?=...) garde le mot-clé dans la partie droite
        split_pattern = rf'\s+(?:et|puis|ensuite|aussi|également)\s+(?:(?:moi|le|la|les|l\'|un|une|des|du)\s+)*(?={action_words})'

        parts = re.split(split_pattern, text, flags=re.IGNORECASE)

        # Nettoyer
        cleaned = []
        for p in parts:
            p = p.strip().rstrip('.!?')
            if len(p) > 3:
                cleaned.append(p)

        if len(cleaned) > 1:
            return cleaned

        return [text]
        
    def _plan(self, objective: str, previous_results: List[Dict]) -> Dict:
        llm_response = ""
        if self.llm_client:
            try:
                context = self._build_context(objective, previous_results)
                llm_response = self._ask_llm(context)
            except Exception as e:
                self.log_callback(f"   ⚠️ LLM planning error : {e}")
                llm_response = ""

        return planner_instance.create_plan(objective, llm_response)

    def _evaluate(self, objective: str, results: List[Dict]) -> Dict:
        if not results:
            return {"complete": False, "reasoning": "Aucun résultat"}

        last = results[-1]

        if last.get("tool") == "conversation":
            return {"complete": True, "reasoning": "Réponse conversationnelle"}

        if last.get("success"):
            return {"complete": True, "reasoning": f"Outil {last.get('tool')} exécuté avec succès"}

        return {"complete": False, "reasoning": f"Échec de {last.get('tool')}, réessai possible"}

    def _build_context(self, objective: str, previous_results: List[Dict]) -> str:
        tools_desc = tool_registry.get_tools_for_prompt()

        context = (
            f"Tu es Micheline, une IA assistante. Tu dois résoudre cette demande :\n"
            f"DEMANDE : {objective}\n\n"
            f"OUTILS DISPONIBLES :\n{tools_desc}\n\n"
        )

        if previous_results:
            context += "RÉSULTATS PRÉCÉDENTS :\n"
            for r in previous_results[-3:]:
                context += f"  - Itération {r['iteration']} [{r['tool']}] : "
                context += f"{'✅' if r['success'] else '❌'} {str(r['result'])[:200]}\n"
            context += "\n"

        context += (
            "Réponds avec l'action à effectuer. Si tu veux utiliser un outil, "
            "formate ta réponse en JSON : {\"tool\": \"nom\", \"params\": {...}}\n"
            "Si c'est une conversation normale, réponds directement."
        )

        return context

    def _ask_llm(self, prompt: str) -> str:
        if not self.llm_client:
            return ""

        try:
            messages = [{"role": "user", "content": prompt}]

            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat(messages)
                if isinstance(response, str):
                    return response
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if hasattr(self.llm_client, 'create_chat_completion'):
                response = self.llm_client.create_chat_completion(messages=messages)
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")

            if callable(self.llm_client):
                response = self.llm_client(prompt)
                if isinstance(response, str):
                    return response

        except Exception as e:
            self.log_callback(f"   ⚠️ LLM error : {e}")

        return ""

    def _build_final_response(self, objective: str, results: List[Dict]) -> str:
        if not results:
            return "Je n'ai pas pu traiter cette demande."

        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        parts = []

        for r in successful:
            result_text = r.get("result", "")
            tool_name = r.get("tool", "")

            # Formatage trading
            if tool_name and tool_name.startswith("trading_"):
                try:
                    from micheline.trading.formatter import format_trading_result, is_trading_tool
                    if is_trading_tool(tool_name):
                        if isinstance(result_text, str):
                            try:
                                import ast
                                result_data = ast.literal_eval(result_text)
                            except (ValueError, SyntaxError):
                                result_data = {"raw": result_text}
                        elif isinstance(result_text, dict):
                            result_data = result_text
                        else:
                            result_data = {"raw": str(result_text)}

                        formatted = format_trading_result(tool_name, result_data)
                        parts.append(formatted)
                        continue
                except ImportError:
                    pass

            # Résultat normal
            if result_text:
                if isinstance(result_text, dict):
                    parts.append(self._format_dict_result(result_text))
                else:
                    parts.append(str(result_text))

        if parts:
            response = "\n\n".join(parts)
        elif failed:
            errors = [
                f"❌ [{r.get('tool', '?')}] {str(r.get('result', 'Erreur'))[:200]}"
                for r in failed
            ]
            response = "Je n'ai pas pu compléter cette tâche :\n" + "\n".join(errors)
        else:
            response = "Tâche traitée mais aucun résultat concret."

        return response

    def _format_dict_result(self, data: dict) -> str:
        if "result" in data and isinstance(data["result"], str):
            return data["result"]

        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"📌 {key}:")
                for k2, v2 in value.items():
                    lines.append(f"   {k2}: {v2}")
            elif isinstance(value, list) and len(value) > 3:
                lines.append(f"📌 {key}: [{len(value)} éléments]")
            else:
                lines.append(f"📌 {key}: {value}")
        return "\n".join(lines)


class AgentPlanner:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._tools_description = ""

    def set_tools_description(self, desc: str):
        self._tools_description = desc


class AgentExecutor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client


class AgentEvaluator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client


class ToolRegistryWrapper:
    def __init__(self):
        self._registry = tool_registry

    def get_tools_description(self) -> str:
        return self._registry.get_tools_for_prompt()

    def list_tools(self) -> dict:
        return self._registry.list_tools()


class AgentBridge:

    AGENT_PATTERNS = [
        r"li[st]\s+le\s+fichier",
        r"[eé]cri[st]\s+dans",
        r"cr[eé]e\s+un\s+fichier",
        r"contenu\s+du\s+(fichier|dossier)",
        r"qu.*(est|y\s*a).*dans\s+le\s+dossier",
        r"info.*syst[eè]me",
        r"combien\s+de\s+(ram|cpu|m[eé]moire|disque)",
        r"calcul[eé]?\s",
        r"quelle?\s+heure",
        r"quel\s+jour",
        r"m[eé]moire",
        r"chemin.*autoris",
        r"ex[eé]cute.*code",
        r"python",
        r"recherch.*web",
        r"actualit[eé]",
        r"command.*syst[eè]me",
        r"mt5",
        r"metatrader",
        r"trading",
        r"backtest",
        r"strat[eé]gi",
        r"d[eé]compos.*probl[eè]me",
        r"ouvr[ei]",
        r"lance",
        r"d[eé]marre",
    ]

    def __init__(self, llm_client=None, log_callback=None, agent_mode=True, **kwargs):
        self.llm_client = llm_client
        self.log_callback = log_callback or (lambda msg: print(f"[AGENT] {msg}"))
        self.agent_mode = agent_mode

        self.agent = AgentLoop(
            llm_client=llm_client,
            log_callback=self.log_callback
        )

        self.tool_registry = ToolRegistryWrapper()

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        tools_count = len(tool_registry.list_tools())
        self.log_callback(f"Bridge initialisé avec {tools_count} outils")

    def process_input(self, user_text: str) -> Dict[str, Any]:
        if not user_text or not user_text.strip():
            return {
                "response": "Message vide.",
                "status": "error",
                "iterations": 0,
                "logs": [],
            }

        try:
            self.log_callback(f"📨 Traitement : {user_text[:80]}...")

            if self.llm_client:
                self.agent.llm_client = self.llm_client
                self.agent.planner.llm_client = self.llm_client
                self.agent.executor.llm_client = self.llm_client
                self.agent.evaluator.llm_client = self.llm_client

            result = self.agent.run(user_text)
            return result

        except Exception as e:
            error_msg = f"Erreur bridge : {type(e).__name__}: {e}"
            self.log_callback(f"❌ {error_msg}")
            traceback.print_exc()
            return {
                "response": f"❌ {error_msg}",
                "status": "error",
                "iterations": 0,
                "logs": [error_msg],
            }

    def detect(self, message: str) -> Tuple[bool, str]:
        if not message:
            return False, "Message vide"

        msg_lower = message.lower().strip()

        for pattern in self.AGENT_PATTERNS:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return True, f"Pattern détecté : {pattern}"

        return False, "Aucun pattern agent"


# Instance globale
agent_bridge = AgentBridge()

# Alias pour compatibilité avec main.py
MichelineBridge = AgentBridge