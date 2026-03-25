# micheline/trading/__init__.py
"""
Module Trading Engine — Phase 6
Génération, évaluation et optimisation de stratégies de trading.
"""

try:
    from micheline.trading.strategy_generator import StrategyGenerator
except ImportError:
    StrategyGenerator = None

try:
    from micheline.trading.metrics import evaluate_result, compute_score
except ImportError:
    evaluate_result = None
    compute_score = None

try:
    from micheline.trading.optimizer import StrategyOptimizer
except ImportError:
    StrategyOptimizer = None

try:
    from micheline.trading.engine import TradingEngine
except ImportError:
    TradingEngine = None

__all__ = [
    "StrategyGenerator",
    "evaluate_result",
    "compute_score",
    "StrategyOptimizer",
    "TradingEngine",
]