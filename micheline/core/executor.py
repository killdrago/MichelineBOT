"""
Executor — Exécute les actions d'un plan une par une.
Utilise le LLM local de Micheline.
"""

from datetime import datetime


class ActionResult:
    """Résultat d'une action exécutée."""

    def __init__(self, action: str, success: bool, output: str, error: str = None):
        self.action = action
        self.success = success
        self.output = output
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "action": self.action,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp
        }

    def __str__(self):
        status = "✅" if self.success else "❌"
        return f"{status} [{self.action}] {self.output[:200]}"


class Executor:
    """
    Exécute les étapes d'un plan.
    Phase 1: actions de base (respond, think)
    Phase 2+: délègue au tool registry
    """

    def __init__(self, llm_client=None, tool_registry=None):
        self.llm_client = llm_client
        self.tool_registry = tool_registry

        self._builtin_actions = {
            "respond": self._action_respond,
            "think": self._action_think,
        }

    def execute_step(self, step: dict) -> ActionResult:
        action_name = step.get("action", "respond")
        params = step.get("params", {})

        try:
            # 1. Actions internes
            if action_name in self._builtin_actions:
                output = self._builtin_actions[action_name](params)
                return ActionResult(action=action_name, success=True, output=output)

            # 2. Tool registry (Phase 2)
            if self.tool_registry:
                output = self.tool_registry.execute(action_name, params)
                return ActionResult(action=action_name, success=True, output=str(output))

            # 3. Action inconnue
            return ActionResult(
                action=action_name, success=False, output="",
                error=f"Action inconnue: {action_name}. Disponibles: {list(self._builtin_actions.keys())}"
            )

        except Exception as e:
            return ActionResult(action=action_name, success=False, output="", error=f"Erreur: {str(e)}")

    def _action_respond(self, params: dict) -> str:
        """Génère une réponse via le LLM local."""
        message = params.get("message", "")

        if self.llm_client and message:
            try:
                if hasattr(self.llm_client, 'chat') and callable(self.llm_client.chat):
                    answer, dt, usage = self.llm_client.chat(
                        messages=[{"role": "user", "content": message}],
                        system_prompt="Tu es Micheline, une IA locale. Réponds de manière utile et concise.",
                        temperature=0.3,
                        max_tokens=1500
                    )
                    return answer or message
            except Exception as e:
                return f"Erreur LLM: {str(e)}"

        return message if message else "Aucun message à transmettre."

    def _action_think(self, params: dict) -> str:
        """Réfléchit/analyse via le LLM local."""
        topic = params.get("topic", params.get("message", ""))

        think_prompt = f"""Analyse cette situation en profondeur.
Donne ton raisonnement étape par étape.

Sujet: {topic}

Réponds avec:
1. Analyse du problème
2. Options possibles
3. Recommandation"""

        if self.llm_client:
            try:
                if hasattr(self.llm_client, 'chat') and callable(self.llm_client.chat):
                    answer, dt, usage = self.llm_client.chat(
                        messages=[{"role": "user", "content": think_prompt}],
                        system_prompt="Tu es un analyste expert. Raisonne étape par étape.",
                        temperature=0.2,
                        max_tokens=1500
                    )
                    return answer or f"Réflexion sur: {topic}"
            except Exception as e:
                return f"Erreur réflexion: {str(e)}"

        return f"Réflexion sur: {topic} (LLM non disponible)"

    def get_available_actions(self) -> list:
        actions = list(self._builtin_actions.keys())
        if self.tool_registry:
            actions.extend(self.tool_registry.list_tools())
        return actions