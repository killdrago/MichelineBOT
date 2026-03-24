"""
Evaluator — Évalue les résultats et décide si l'objectif est atteint.
Utilise le LLM local de Micheline.
"""

import json


class Evaluation:
    """Résultat d'une évaluation."""

    def __init__(self, is_complete: bool, satisfaction: float, summary: str, next_action: str = None):
        self.is_complete = is_complete
        self.satisfaction = satisfaction
        self.summary = summary
        self.next_action = next_action

    def __str__(self):
        status = "✅ COMPLET" if self.is_complete else "🔄 EN COURS"
        return f"{status} | Satisfaction: {self.satisfaction:.0%} | {self.summary}"


class Evaluator:
    """Évalue si l'objectif est atteint après exécution des actions."""

    EVAL_PROMPT = """Tu es Micheline. Tu évalues si un objectif a été atteint.

OBJECTIF INITIAL: {objective}

ACTIONS EXÉCUTÉES:
{actions_summary}

RÉSULTATS OBTENUS:
{results_summary}

Réponds UNIQUEMENT en JSON:
{{
    "is_complete": true ou false,
    "satisfaction": 0.0 à 1.0,
    "summary": "résumé en 1-2 phrases",
    "next_action": "continue" ou "retry" ou "modify_plan" ou "abort" ou null,
    "reasoning": "pourquoi cette évaluation"
}}"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def evaluate(self, objective: str, results: list) -> Evaluation:
        actions_summary = ""
        results_summary = ""
        all_success = True

        for r in results:
            status = "✅" if r.success else "❌"
            actions_summary += f"  {status} {r.action}\n"
            if r.success:
                results_summary += f"  [{r.action}]: {r.output[:300]}\n"
            else:
                results_summary += f"  [{r.action}] ERREUR: {r.error}\n"
                all_success = False

        if self.llm_client:
            return self._evaluate_with_llm(objective, actions_summary, results_summary)

        # Fallback sans LLM
        if all_success:
            return Evaluation(is_complete=True, satisfaction=0.8, summary="Toutes les actions ont réussi.", next_action=None)
        else:
            failed_count = sum(1 for r in results if not r.success)
            return Evaluation(
                is_complete=False,
                satisfaction=max(0, 1 - (failed_count / max(len(results), 1))),
                summary=f"{failed_count} action(s) ont échoué.",
                next_action="retry"
            )

    def _evaluate_with_llm(self, objective: str, actions_summary: str, results_summary: str) -> Evaluation:
        prompt = self.EVAL_PROMPT.format(
            objective=objective,
            actions_summary=actions_summary,
            results_summary=results_summary
        )

        try:
            if hasattr(self.llm_client, 'chat') and callable(self.llm_client.chat):
                answer, dt, usage = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="Tu es un évaluateur. Réponds UNIQUEMENT en JSON valide.",
                    temperature=0.1,
                    max_tokens=800
                )
                text = answer or ""
            else:
                raise ValueError("Client LLM non reconnu")

            # Parser le JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                return Evaluation(
                    is_complete=data.get("is_complete", False),
                    satisfaction=data.get("satisfaction", 0.5),
                    summary=data.get("summary", "Évaluation terminée"),
                    next_action=data.get("next_action")
                )

        except Exception as e:
            print(f"[Evaluator] Erreur: {e}")

        return Evaluation(
            is_complete=True,
            satisfaction=0.5,
            summary="Évaluation par défaut (erreur parsing)",
            next_action=None
        )