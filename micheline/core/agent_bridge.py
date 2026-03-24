"""
Agent Bridge — Pont entre main.py et le système agent.
Emplacement : micheline/core/agent_bridge.py
COMPATIBLE avec main.py (MichelineBridge)
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
        """
        Exécute la boucle agent complète.

        Returns:
            {
                "response": str,
                "status": "success" | "partial" | "error",
                "iterations": int,
                "logs": [str]
            }
        """
        logs = []
        all_results = []
        status = "error"

        self.log_callback(f"🎯 Objectif : {objective}")
        logs.append(f"🎯 Objectif : {objective}")

        for i in range(self.max_iterations):
            self.log_callback(f"━━━ Itération {i + 1}/{self.max_iterations} ━━━")
            logs.append(f"━━━ Itération {i + 1}/{self.max_iterations} ━━━")

            try:
                # PLAN
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

                # EXECUTE
                self.log_callback("⚡ Exécution...")
                logs.append("⚡ Exécution...")

                exec_result = executor_instance.execute(plan)
                result_text = exec_result.get("result", "")
                success = exec_result.get("success", False)

                self.log_callback(f"   Succès : {success}")
                self.log_callback(f"   Résultat : {str(result_text)[:300]}")
                logs.append(f"   Succès : {success} | Résultat : {str(result_text)[:300]}")

                all_results.append({
                    "iteration": i + 1,
                    "tool": tool_name,
                    "success": success,
                    "result": str(result_text)[:1000]
                })

                # EVALUATE
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
                    "result": str(e)
                })
                continue

        # Construire la réponse finale
        if status != "success" and all_results:
            status = "partial"

        response = self._build_final_response(objective, all_results)

        return {
            "response": response,
            "status": status,
            "iterations": len(all_results),
            "logs": logs
        }

    def _plan(self, objective: str, previous_results: List[Dict]) -> Dict:
        """Planifie la prochaine action."""
        # Obtenir une réponse du LLM pour guider la planification
        llm_response = ""
        if self.llm_client:
            try:
                context = self._build_context(objective, previous_results)
                llm_response = self._ask_llm(context)
            except Exception as e:
                self.log_callback(f"   ⚠️ LLM planning error : {e}")
                llm_response = ""

        # Utiliser le planner pour créer le plan
        return planner_instance.create_plan(objective, llm_response)

    def _evaluate(self, objective: str, results: List[Dict]) -> Dict:
        """Évalue si l'objectif est atteint."""
        if not results:
            return {"complete": False, "reasoning": "Aucun résultat"}

        last = results[-1]

        # Si le dernier résultat est une conversation, c'est terminé
        if last.get("tool") == "conversation":
            return {"complete": True, "reasoning": "Réponse conversationnelle"}

        # Si succès sur un outil, considérer comme terminé
        # (pour la v1, on fait simple — 1 itération = terminé si succès)
        if last.get("success"):
            return {"complete": True, "reasoning": f"Outil {last.get('tool')} exécuté avec succès"}

        # Si échec, continuer seulement s'il reste des itérations
        return {"complete": False, "reasoning": f"Échec de {last.get('tool')}, réessai possible"}

    def _build_context(self, objective: str, previous_results: List[Dict]) -> str:
        """Construit le contexte pour le LLM."""
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
                context += f"{'✅' if r['success'] else '❌'} {r['result'][:200]}\n"
            context += "\n"

        context += (
            "Réponds avec l'action à effectuer. Si tu veux utiliser un outil, "
            "formate ta réponse en JSON : {\"tool\": \"nom\", \"params\": {...}}\n"
            "Si c'est une conversation normale, réponds directement."
        )

        return context

    def _ask_llm(self, prompt: str) -> str:
        """Interroge le LLM."""
        if not self.llm_client:
            return ""

        try:
            messages = [{"role": "user", "content": prompt}]

            # Essayer la méthode .chat() (llama-cpp-python)
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat(messages)
                if isinstance(response, str):
                    return response
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Essayer .create_chat_completion()
            if hasattr(self.llm_client, 'create_chat_completion'):
                response = self.llm_client.create_chat_completion(messages=messages)
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Essayer .__call__()
            if callable(self.llm_client):
                response = self.llm_client(prompt)
                if isinstance(response, str):
                    return response

        except Exception as e:
            self.log_callback(f"   ⚠️ LLM error : {e}")

        return ""

    def _build_final_response(self, objective: str, results: List[Dict]) -> str:
        """Construit la réponse finale à partir de tous les résultats."""
        if not results:
            return "Je n'ai pas pu traiter cette demande."

        # Collecter les résultats réussis
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        parts = []

        for r in successful:
            result_text = r.get("result", "")
            if result_text:
                parts.append(result_text)

        if parts:
            response = "\n\n".join(parts)
        elif failed:
            errors = [f"❌ [{r.get('tool', '?')}] {r.get('result', 'Erreur inconnue')[:200]}" for r in failed]
            response = "Je n'ai pas pu compléter cette tâche :\n" + "\n".join(errors)
        else:
            response = "Tâche traitée mais aucun résultat concret."

        return response


class AgentPlanner:
    """Wrapper planner pour compatibilité avec main.py."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._tools_description = ""

    def set_tools_description(self, desc: str):
        """Met à jour la description des outils pour le prompt."""
        self._tools_description = desc


class AgentExecutor:
    """Wrapper executor pour compatibilité avec main.py."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client


class AgentEvaluator:
    """Wrapper evaluator pour compatibilité avec main.py."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client


class ToolRegistryWrapper:
    """Wrapper pour le tool_registry compatible avec main.py."""

    def __init__(self):
        self._registry = tool_registry

    def get_tools_description(self) -> str:
        """Retourne la description des outils (appelé par main.py)."""
        return self._registry.get_tools_for_prompt()

    def list_tools(self) -> dict:
        """Retourne la liste des outils (appelé par main.py)."""
        return self._registry.tools


class AgentBridge:
    """
    Pont principal entre main.py et le système agent.
    Compatible avec l'ancien MichelineBridge.
    """

    # Patterns qui déclenchent l'agent (utilisés par _detect_agent_mode dans main.py)
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
        r"d[eé]compos.*probl[eè]me",
    ]

    def __init__(self, llm_client=None, log_callback=None, agent_mode=True, **kwargs):
        """
        Initialise le bridge.

        Args:
            llm_client: Instance du LLM local
            log_callback: Fonction de logging vers la console
            agent_mode: Active le mode agent (toujours True)
        """
        self.llm_client = llm_client
        self.log_callback = log_callback or (lambda msg: print(f"[AGENT] {msg}"))
        self.agent_mode = agent_mode

        # Créer l'agent loop
        self.agent = AgentLoop(
            llm_client=llm_client,
            log_callback=self.log_callback
        )

        # Créer le wrapper du tool registry
        self.tool_registry = ToolRegistryWrapper()

        # Stocker tout argument supplémentaire
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        self.log_callback(f"Bridge initialisé avec {len(self.tool_registry.list_tools())} outils")

    def process_input(self, user_text: str) -> Dict[str, Any]:
        """
        Traite une entrée utilisateur via le système agent.

        Args:
            user_text: Le message de l'utilisateur

        Returns:
            {
                "response": str,
                "status": "success" | "partial" | "error",
                "iterations": int,
                "logs": [str]
            }
        """
        if not user_text or not user_text.strip():
            return {
                "response": "Message vide.",
                "status": "error",
                "iterations": 0,
                "logs": []
            }

        try:
            self.log_callback(f"📨 Traitement : {user_text[:80]}...")

            # Mettre à jour le LLM dans l'agent si nécessaire
            if self.llm_client:
                self.agent.llm_client = self.llm_client
                self.agent.planner.llm_client = self.llm_client
                self.agent.executor.llm_client = self.llm_client
                self.agent.evaluator.llm_client = self.llm_client

            # Lancer la boucle agent
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
                "logs": [error_msg]
            }

    def detect(self, message: str) -> Tuple[bool, str]:
        """
        Détecte si un message est une demande agent.
        Note : main.py utilise sa propre méthode _detect_agent_mode,
        mais cette méthode est là pour compatibilité si appelée directement.
        """
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