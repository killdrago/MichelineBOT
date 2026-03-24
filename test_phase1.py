"""Test rapide de la Phase 1 — fonctionne SANS client LLM."""

import sys
sys.path.insert(0, ".")

from micheline.core.agent_bridge import MichelineBridge


def test_sans_llm():
    """Test sans LLM — vérifier que la structure fonctionne."""
    print("=" * 60)
    print("TEST PHASE 1 — Sans LLM")
    print("=" * 60)

    bridge = MichelineBridge(
        llm_client=None,  # Pas de LLM
        log_callback=lambda msg: print(f"  📝 {msg}"),
        agent_mode=True
    )

    # Test 1: Message simple
    print("\n--- Test 1: Message simple ---")
    result = bridge.process_input("Bonjour Micheline!")
    print(f"Réponse: {result['response']}")
    print(f"Status: {result['status']}")

    # Test 2: Question
    print("\n--- Test 2: Question ---")
    result = bridge.process_input("Comment vas-tu?")
    print(f"Réponse: {result['response']}")

    # Test 3: Objectif complexe (sans LLM, fallback)
    print("\n--- Test 3: Objectif complexe ---")
    result = bridge.process_input("Analyse les performances du marché EURUSD cette semaine")
    print(f"Réponse: {result['response'][:200]}")
    print(f"Itérations: {result['iterations']}")

    # Test 4: État de l'agent
    print("\n--- Test 4: État agent ---")
    state = bridge.get_state()
    print(f"État: {state}")

    print("\n" + "=" * 60)
    print("✅ Phase 1 fonctionnelle!")
    print("=" * 60)


if __name__ == "__main__":
    test_sans_llm()