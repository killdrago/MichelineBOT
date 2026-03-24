"""
Bridge — Fait le pont entre l'interface existante et l'agent.
Initialise le Tool Registry (Phase 2) et connecte tout.
"""

from typing import Callable
from .agent_loop import AgentLoop

# Import du système d'outils (Phase 2)
try:
    from micheline.tools.registry import ToolRegistry
    from micheline.tools.system_tools import register_system_tools
    from micheline.tools.file_tools import register_file_tools
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"[AgentBridge] Tools non disponibles: {e}")
    TOOLS_AVAILABLE = False


class MichelineBridge:
    """
    Interface simple entre l'app existante et le système agent.
    Initialise automatiquement le Tool Registry.
    """

    def __init__(self, llm_client=None, log_callback: Callable = None, agent_mode: bool = True):
        self.llm_client = llm_client
        self.log_callback = log_callback or print
        self.agent_mode = agent_mode
        self.logs = []

        # === Phase 2: Initialiser le Tool Registry ===
        self.tool_registry = None
        if TOOLS_AVAILABLE:
            try:
                self.tool_registry = ToolRegistry()
                register_system_tools(self.tool_registry)
                register_file_tools(self.tool_registry)
                print(f"[AgentBridge] ✅ Tool Registry initialisé avec {len(self.tool_registry.list_tools())} outils")
            except Exception as e:
                print(f"[AgentBridge] ⚠️ Erreur init tools: {e}")
                self.tool_registry = None

        # Créer l'agent avec le registry
        self.agent = AgentLoop(
            llm_client=llm_client,
            tool_registry=self.tool_registry,
            on_update=self._on_agent_update
        )

        # Donner la description des outils au Planner
        if self.tool_registry:
            tools_desc = self.tool_registry.get_tools_description()
            self.agent.planner.set_tools_description(tools_desc)
            print(f"[AgentBridge] Planner informé de {len(self.tool_registry.list_tools())} outils")
            
    def process_input(self, user_message: str) -> dict:
        self.logs = []

        if not user_message.strip():
            return {
                "response": "Je n'ai pas reçu de message.",
                "logs": [],
                "status": "error",
                "iterations": 0
            }

        try:
            if self.agent_mode:
                response = self.agent.quick_respond(user_message)
            else:
                response = self.agent.executor._action_respond({"message": user_message})

            return {
                "response": response,
                "logs": self.logs.copy(),
                "status": self.agent.state.status,
                "iterations": self.agent.state.iteration
            }

        except Exception as e:
            return {
                "response": f"Erreur: {str(e)}",
                "logs": self.logs.copy(),
                "status": "error",
                "iterations": 0
            }

    def toggle_agent_mode(self, enabled: bool):
        self.agent_mode = enabled

    def get_state(self) -> dict:
        return {
            "status": self.agent.state.status,
            "iteration": self.agent.state.iteration,
            "objective": self.agent.state.objective,
            "history_length": len(self.agent.state.history),
            "tools_available": self.tool_registry.list_tools() if self.tool_registry else []
        }

    def _on_agent_update(self, message: str, state):
        self.logs.append(message)
        if self.log_callback:
            self.log_callback(message)