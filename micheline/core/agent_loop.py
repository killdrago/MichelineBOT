"""
Agent Loop — Boucle principale de l'agent Micheline.
Emplacement : micheline/core/agent_loop.py
FICHIER MODIFIÉ — Correction multi-actions
"""

import os
import sys
import re
import time
import traceback
from typing import List, Dict, Any, Optional

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.planner import planner
from core.executor import executor
from core.evaluator import evaluator

try:
    from memory.memory import memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    memory_manager = None

try:
    from langchain_ollama import OllamaLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class AgentLoop:
    """Boucle principale de l'agent autonome Micheline."""

    def __init__(self):
        self.running = False
        self.history = []
        self.llm = None
        self.iteration_count = 0
        self.max_iterations = 100

        if LLM_AVAILABLE:
            try:
                self.llm = OllamaLLM(
                    model="llama3.2",
                    base_url="http://localhost:11434",
                    temperature=0.7
                )
                print("✅ LLM Ollama connecté")
            except Exception as e:
                print(f"⚠️ LLM non disponible: {e}")

    def run(self):
        """Boucle principale interactive."""
        self.running = True
        print("\n" + "=" * 60)
        print("🤖 MICHELINE v3 — Agent Autonome")
        print("=" * 60)
        print("Commandes: 'quit' pour quitter, 'status' pour l'état")
        print("=" * 60 + "\n")

        while self.running:
            try:
                objective = input("\n🎯 Objectif > ").strip()

                if not objective:
                    continue

                if objective.lower() in ('quit', 'exit', 'q'):
                    print("\n👋 Arrêt de Micheline...")
                    self.running = False
                    break

                if objective.lower() == 'status':
                    self._show_status()
                    continue

                if objective.lower() == 'help':
                    self._show_help()
                    continue

                # ═══════════════════════════════════════════════
                # TRAITEMENT : découper en sous-objectifs
                # ═══════════════════════════════════════════════
                from core.planner import planner
                sub_objectives = planner.split_objectives(objective)

                if len(sub_objectives) > 1:
                    print(f"\n📋 {len(sub_objectives)} actions détectées :")
                    for i, sub in enumerate(sub_objectives, 1):
                        print(f"   {i}. {sub}")
                    print()

                # Exécuter CHAQUE sous-objectif indépendamment
                all_results = []
                for idx, sub_obj in enumerate(sub_objectives):
                    if len(sub_objectives) > 1:
                        print(f"\n{'─' * 50}")
                        print(f"▶ Action {idx + 1}/{len(sub_objectives)} : {sub_obj}")
                        print(f"{'─' * 50}")

                    result = self._process_single_objective(sub_obj)
                    all_results.append(result)

                # Résumé si multi-actions
                if len(sub_objectives) > 1:
                    print(f"\n{'═' * 50}")
                    print(f"📊 RÉSUMÉ — {len(all_results)} actions exécutées :")
                    for i, r in enumerate(all_results, 1):
                        status = "✅" if r.get("success", False) else "❌"
                        tool = r.get("tool_used", "inconnu")
                        print(f"   {status} Action {i}: {tool}")
                    print(f"{'═' * 50}")

            except KeyboardInterrupt:
                print("\n\n⚠️ Interruption clavier")
                confirm = input("Quitter ? (o/n) > ").strip().lower()
                if confirm in ('o', 'oui', 'y', 'yes'):
                    self.running = False
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                traceback.print_exc()

    def _process_single_objective(self, objective: str) -> Dict[str, Any]:
        """
        Traite UN SEUL objectif de bout en bout.
        Retourne le résultat de l'exécution.
        """
        self.iteration_count += 1
        result_info = {
            "objective": objective,
            "success": False,
            "tool_used": "none"
        }

        try:
            # ═══════════════════════════════════════
            # ÉTAPE 1 : Réflexion (LLM)
            # ═══════════════════════════════════════
            llm_response = self._think(objective)

            # ═══════════════════════════════════════
            # ÉTAPE 2 : Planification
            # ═══════════════════════════════════════
            plan = planner.create_plan(objective, llm_response)
            tool_name = plan.get("tool", "conversation")
            params = plan.get("params", {})
            reasoning = plan.get("reasoning", "")

            print(f"\n🔧 Outil: {tool_name}")
            if reasoning:
                print(f"💭 Raison: {reasoning}")

            result_info["tool_used"] = tool_name

            # ═══════════════════════════════════════
            # ÉTAPE 3 : Exécution
            # ═══════════════════════════════════════
            if tool_name == "conversation":
                response = params.get("response", llm_response)
                print(f"\n💬 {response}")
                result_info["success"] = True

            else:
                result = executor.execute(tool_name, params)

                # ═══════════════════════════════════
                # ÉTAPE 4 : Évaluation
                # ═══════════════════════════════════
                evaluation = evaluator.evaluate(
                    objective=objective,
                    tool_used=tool_name,
                    result=result
                )

                self._display_result(tool_name, result, evaluation)
                result_info["success"] = evaluation.get("success", False)

                # ═══════════════════════════════════
                # ÉTAPE 5 : Mémorisation
                # ═══════════════════════════════════
                if MEMORY_AVAILABLE and memory_manager:
                    try:
                        memory_manager.store({
                            "type": "action",
                            "objective": objective,
                            "tool": tool_name,
                            "success": evaluation.get("success", False),
                            "score": evaluation.get("score", 0),
                            "iteration": self.iteration_count
                        })
                    except Exception:
                        pass

            # Historique
            self.history.append({
                "iteration": self.iteration_count,
                "objective": objective,
                "tool": tool_name,
                "success": result_info["success"]
            })

        except Exception as e:
            print(f"\n❌ Erreur lors du traitement: {e}")
            traceback.print_exc()
            result_info["success"] = False

        return result_info

    def _think(self, objective: str) -> str:
        """Utilise le LLM pour réfléchir à l'objectif."""
        if not self.llm:
            return ""

        prompt = f"""Tu es Micheline, une IA assistante. L'utilisateur demande :
"{objective}"

Si c'est une question de conversation, réponds directement.
Si ça nécessite un outil, indique lequel en JSON :
{{"tool": "nom_outil", "params": {{}}, "reasoning": "pourquoi"}}

Outils disponibles : calculator, datetime, system_info, list_directory,
read_file, write_file, file_info, memory_search, memory_stats,
list_allowed_paths, code_executor, web_search, shell_command,
mt5_tool, trading_search, trading_quick_test, trading_improve,
trading_report, trading_top_strategies, task_planner, app_launcher

Réponds de manière concise."""

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"⚠️ Erreur LLM: {e}")
            return ""

    def _display_result(self, tool_name: str, result: Dict, evaluation: Dict):
        """Affiche le résultat de manière formatée."""
        success = result.get("success", False)
        status = "✅" if success else "❌"

        print(f"\n{status} Résultat [{tool_name}]:")

        # Affichage spécifique par outil
        if tool_name == "app_launcher":
            output = result.get("output", result.get("result", ""))
            if isinstance(output, str):
                print(f"   {output}")
            elif isinstance(output, dict):
                for key, val in output.items():
                    print(f"   {key}: {val}")

        elif tool_name in ("trading_search", "trading_quick_test",
                           "trading_improve", "trading_report",
                           "trading_top_strategies"):
            output = result.get("output", result.get("result", ""))
            if isinstance(output, str):
                # Le trading engine produit souvent du texte formaté
                print(output)
            elif isinstance(output, dict):
                for key, val in output.items():
                    if key not in ("success",):
                        print(f"   {key}: {val}")

        else:
            output = result.get("output", result.get("result", ""))
            if isinstance(output, str):
                # Limiter l'affichage
                if len(output) > 2000:
                    print(f"   {output[:2000]}...")
                else:
                    print(f"   {output}")
            elif isinstance(output, dict):
                for key, val in output.items():
                    print(f"   {key}: {val}")
            elif isinstance(output, list):
                for item in output[:20]:
                    print(f"   • {item}")
                if len(output) > 20:
                    print(f"   ... et {len(output) - 20} de plus")

        # Score d'évaluation
        score = evaluation.get("score", 0)
        if score > 0:
            print(f"\n   📊 Score: {score}/100")

    def _show_status(self):
        """Affiche l'état actuel de l'agent."""
        print("\n" + "=" * 40)
        print("📊 STATUS MICHELINE")
        print("=" * 40)
        print(f"   Itérations: {self.iteration_count}")
        print(f"   LLM: {'✅ Connecté' if self.llm else '❌ Non disponible'}")
        print(f"   Mémoire: {'✅ Active' if MEMORY_AVAILABLE else '❌ Non disponible'}")

        if self.history:
            success_count = sum(1 for h in self.history if h.get("success"))
            total = len(self.history)
            print(f"   Succès: {success_count}/{total} ({100*success_count//total}%)")

            # Dernières actions
            print("\n   Dernières actions:")
            for h in self.history[-5:]:
                s = "✅" if h["success"] else "❌"
                print(f"   {s} [{h['tool']}] {h['objective'][:50]}")

        print("=" * 40)

    def _show_help(self):
        """Affiche l'aide."""
        print("""
╔══════════════════════════════════════════════╗
║           🤖 AIDE MICHELINE v3               ║
╠══════════════════════════════════════════════╣
║                                              ║
║  💬 Conversation normale                     ║
║  🔢 "calcule 15 * 37"                        ║
║  📁 "liste le dossier configs"               ║
║  🕐 "quelle heure est-il ?"                  ║
║  💻 "info système"                            ║
║  🐍 "exécute print('hello')"                 ║
║  🌐 "recherche sur Wikipedia: Python"         ║
║  📂 "ouvre paint" / "ouvre notepad"            ║
║                                              ║
║  📈 TRADING:                                  ║
║  "cherche stratégie trading EURUSD"           ║
║  "test rapide 10 stratégies"                  ║
║  "améliore la stratégie"                      ║
║  "rapport trading"                            ║
║  "top 5 stratégies"                           ║
║                                              ║
║  🔀 MULTI-ACTIONS:                            ║
║  "calcule 2+2 et ouvre paint"                 ║
║  "cherche stratégie et ouvre notepad"          ║
║                                              ║
║  ⚙️ Commandes:                                ║
║  'status' — état de l'agent                   ║
║  'help'   — cette aide                        ║
║  'quit'   — quitter                           ║
║                                              ║
╚══════════════════════════════════════════════╝
""")


def main():
    """Point d'entrée principal."""
    agent = AgentLoop()
    agent.run()


if __name__ == "__main__":
    main()