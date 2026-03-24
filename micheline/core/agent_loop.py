"""
Agent Loop — Le cœur de Micheline v3.
Boucle autonome: Objectif → Plan → Exécution → Évaluation → Boucle

C'est CE fichier qui est appelé depuis ton onglet Interaction.
"""

import time
from datetime import datetime
from typing import Callable, Optional

from .planner import Planner, Plan
from .executor import Executor, ActionResult
from .evaluator import Evaluator, Evaluation

# Mémoire persistante (Phase 4)
try:
    from micheline.memory.memory import AgentMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    AgentMemory = None


class AgentState:
    """État courant de l'agent."""

    def __init__(self):
        self.objective = ""
        self.current_plan = None
        self.history = []  # Liste de {"step", "result", "timestamp"}
        self.iteration = 0
        self.max_iterations = 10  # Sécurité anti-boucle infinie
        self.status = "idle"  # idle, thinking, executing, evaluating, complete, error
        self.final_output = ""

    def reset(self, objective: str):
        self.objective = objective
        self.current_plan = None
        self.history = []
        self.iteration = 0
        self.status = "thinking"
        self.final_output = ""


class AgentLoop:
    """
    Boucle agent autonome.

    Usage depuis ton onglet Interaction:
        agent = AgentLoop(llm_client=ton_client)
        result = agent.run("Trouve une stratégie rentable sur EURUSD")
    """

    def __init__(self, llm_client=None, tool_registry=None, on_update: Callable = None):
        self.planner = Planner(llm_client=llm_client)
        self.executor = Executor(llm_client=llm_client, tool_registry=tool_registry)
        self.evaluator = Evaluator(llm_client=llm_client)

        self.state = AgentState()
        self.on_update = on_update or (lambda msg, state: None)
        self.llm_client = llm_client

        # === Phase 4: Mémoire persistante ===
        self.memory = None
        if MEMORY_AVAILABLE:
            try:
                self.memory = AgentMemory()
            except Exception as e:
                print(f"[AgentLoop] Mémoire non disponible: {e}")
        """
        Args:
            llm_client: ton client LLM existant (Claude/OpenAI)
            tool_registry: registre des outils (Phase 2, None pour l'instant)
            on_update: callback appelé à chaque étape pour mettre à jour l'UI
                       signature: on_update(message: str, state: AgentState)
        """
        self.planner = Planner(llm_client=llm_client)
        self.executor = Executor(llm_client=llm_client, tool_registry=tool_registry)
        self.evaluator = Evaluator(llm_client=llm_client)

        self.state = AgentState()
        self.on_update = on_update or (lambda msg, state: None)
        self.llm_client = llm_client

    def run(self, objective: str) -> str:
        """
        Point d'entrée principal. Appelé depuis l'onglet Interaction.

        Args:
            objective: la demande de l'utilisateur (texte brut)

        Returns:
            str: la réponse finale à afficher
        """
        self.state.reset(objective)
        self._emit(f"🎯 Objectif reçu: {objective}")

        try:
            while self.state.iteration < self.state.max_iterations:
                self.state.iteration += 1
                self._emit(f"\n{'='*50}")
                self._emit(f"🔄 Itération {self.state.iteration}/{self.state.max_iterations}")

                # === ÉTAPE 1: PLANIFIER ===
                self.state.status = "thinking"
                self._emit("🧠 Réflexion et planification...")

                context = self._build_context()
                available_tools = self.executor.get_available_actions()

                plan = self.planner.create_plan(
                    objective=objective,
                    context=context,
                    available_tools=available_tools
                )
                self.state.current_plan = plan
                self._emit(str(plan))

                # === ÉTAPE 2: EXÉCUTER ===
                self.state.status = "executing"
                results = []

                while not plan.is_complete():
                    step = plan.next_step()
                    if step is None:
                        break

                    self._emit(f"⚡ Exécution: {step.get('description', step.get('action', '?'))}")
                    result = self.executor.execute_step(step)
                    results.append(result)

                    self.state.history.append({
                        "iteration": self.state.iteration,
                        "step": step,
                        "result": result.to_dict(),
                        "timestamp": datetime.now().isoformat()
                    })

                    self._emit(str(result))

                    # === Phase 4: Stocker en mémoire persistante ===
                    if self.memory:
                        try:
                            self.memory.store_experience(
                                objective=objective,
                                action=result.action,
                                params=step.get("params", {}),
                                result=result.output[:2000] if result.output else "",
                                success=result.success,
                                notes=result.error if not result.success else None
                            )
                        except Exception as e:
                            print(f"[AgentLoop] Erreur stockage mémoire: {e}")
                            
                    # Si erreur critique, on arrête cette itération
                    if not result.success:
                        self._emit(f"⚠️ Erreur: {result.error}")
                        break

                # === ÉTAPE 3: ÉVALUER ===
                self.state.status = "evaluating"
                self._emit("📊 Évaluation des résultats...")

                evaluation = self.evaluator.evaluate(objective, results)
                self._emit(str(evaluation))

                # === DÉCISION ===
                if evaluation.is_complete:
                    self.state.status = "complete"
                    self.state.final_output = self._compile_final_output(results)
                    self._emit(f"\n✅ Objectif atteint! (satisfaction: {evaluation.satisfaction:.0%})")
                    return self.state.final_output

                elif evaluation.next_action == "abort":
                    self.state.status = "error"
                    self._emit("🛑 Abandon: objectif jugé impossible.")
                    return f"❌ Impossible d'atteindre l'objectif: {evaluation.summary}"

                elif evaluation.next_action in ("retry", "modify_plan", "continue"):
                    self._emit(f"🔄 {evaluation.next_action}: on continue...")
                    # La boucle continue avec le contexte enrichi
                    continue

                else:
                    # Par défaut, si évaluation ambiguë, on considère terminé
                    self.state.status = "complete"
                    self.state.final_output = self._compile_final_output(results)
                    return self.state.final_output

            # Max itérations atteint
            self.state.status = "error"
            self._emit(f"⚠️ Maximum d'itérations atteint ({self.state.max_iterations})")
            return self._compile_final_output(
                [r for h in self.state.history for r in [ActionResult(**h["result"])] if True]
            ) or "⚠️ Nombre maximum d'itérations atteint sans résolution complète."

        except Exception as e:
            self.state.status = "error"
            error_msg = f"❌ Erreur agent: {str(e)}"
            self._emit(error_msg)
            return error_msg

    def _build_context(self) -> str:
        """Construit le contexte à partir de l'historique + mémoire persistante."""
        parts = []

        # Contexte de la session en cours
        if self.state.history:
            parts.append("=== Session en cours ===")
            for entry in self.state.history[-5:]:
                result = entry["result"]
                status = "OK" if result["success"] else "ERREUR"
                parts.append(
                    f"[Iter {entry['iteration']}] {result['action']}: {status} — {result['output'][:200]}"
                )

        # Contexte de la mémoire persistante (Phase 4)
        if self.memory:
            try:
                mem_context = self.memory.get_context_summary(max_items=5)
                if mem_context and mem_context != "Aucune mémoire enregistrée.":
                    parts.append("\n" + mem_context)
            except Exception as e:
                print(f"[AgentLoop] Erreur lecture mémoire: {e}")

        return "\n".join(parts) if parts else ""
        
    def _compile_final_output(self, results: list) -> str:
        """Compile la sortie finale à partir des résultats."""
        # Prendre la dernière réponse réussie
        for result in reversed(results):
            if isinstance(result, ActionResult):
                if result.success and result.output:
                    return result.output
            elif isinstance(result, dict):
                if result.get("success") and result.get("output"):
                    return result["output"]

        return "Traitement terminé mais aucun résultat concret obtenu."

    def _emit(self, message: str):
        """Envoie un message de mise à jour."""
        self.on_update(message, self.state)

    # === MÉTHODE SIMPLE POUR QUESTIONS DIRECTES ===
    def quick_respond(self, message: str) -> str:
        """
        Point d'entrée depuis le bridge.
        Comme le routing agent/conversation est déjà fait dans main.py,
        ici on lance TOUJOURS la boucle agent complète.
        """
        # Toujours utiliser la boucle agent complète
        # (la détection simple/complexe est déjà faite dans main.py)
        return self.run(message)