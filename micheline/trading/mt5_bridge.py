"""
micheline/trading/mt5_bridge.py

Communication avec le Bridge EA dans MT5.
Envoie les stratégies à tester, attend les résultats.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("micheline.trading.mt5_bridge")


class MT5Bridge:
    """
    Communication Python ↔ MT5 via fichiers JSON dans le Common folder.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.common_path = self._find_common_path()
        self.bridge_folder = None

        if self.common_path:
            self.bridge_folder = os.path.join(self.common_path, "Micheline")
            os.makedirs(self.bridge_folder, exist_ok=True)

        self._available = self.bridge_folder is not None
        logger.info(f"MT5 Bridge: {'✅ OK' if self._available else '❌ Non disponible'}")
        if self._available:
            logger.info(f"  Dossier: {self.bridge_folder}")

    def _find_common_path(self) -> Optional[str]:
        """Trouve le dossier Common de MT5."""
        common = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files")
        if os.path.exists(common):
            return common

        # Chercher dans les sous-dossiers
        mq_path = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal")
        if os.path.exists(mq_path):
            for item in os.listdir(mq_path):
                candidate = os.path.join(mq_path, item, "MQL5", "Files")
                if os.path.exists(candidate):
                    return candidate

        return None

    def is_available(self) -> bool:
        """Vérifie si le bridge est disponible."""
        if not self._available:
            return False
        # Vérifier si l'EA est actif
        status = self._read_status()
        return status is not None and status.get("status") in ("ready", "processing")

    def is_ea_running(self) -> bool:
        """Vérifie si le Bridge EA tourne dans MT5."""
        status = self._read_status()
        if status is None:
            return False
        # L'EA est considéré actif si le status a été mis à jour récemment
        return status.get("status") in ("ready", "processing")

    def run_backtest(self, strategy: dict) -> Optional[Dict[str, Any]]:
        """
        Envoie une stratégie au Bridge EA et attend les résultats.

        Args:
            strategy: dict de la stratégie complète

        Returns:
            dict avec les résultats du backtest ou None si timeout
        """
        if not self._available:
            logger.error("Bridge non disponible")
            return None

        # Convertir la stratégie en commande JSON
        command = self._strategy_to_command(strategy)

        # Écrire la commande
        command_file = os.path.join(self.bridge_folder, "command.json")
        result_file = os.path.join(self.bridge_folder, "result.json")

        # Supprimer l'ancien résultat
        if os.path.exists(result_file):
            try:
                os.remove(result_file)
            except Exception:
                pass

        # Écrire la commande
        try:
            with open(command_file, "w", encoding="utf-8") as f:
                json.dump(command, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur écriture commande: {e}")
            return None

        logger.info(f"📤 Commande envoyée: {strategy.get('id', '?')}")

        # Attendre le résultat
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if os.path.exists(result_file):
                try:
                    # Attendre un peu que l'écriture soit finie
                    time.sleep(0.2)
                    with open(result_file, "r", encoding="utf-8") as f:
                        result = json.load(f)

                    # Vérifier que c'est bien pour notre stratégie
                    if result.get("strategy_id") == strategy.get("id", ""):
                        logger.info(f"📥 Résultat reçu: {result.get('profit_money', 0):.2f}$")
                        # Nettoyer
                        try:
                            os.remove(result_file)
                        except Exception:
                            pass
                        return self._parse_result(result)

                except json.JSONDecodeError:
                    # Fichier pas encore complet
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning(f"Erreur lecture résultat: {e}")
                    time.sleep(0.3)

            time.sleep(0.5)

        logger.error(f"⏱️ Timeout ({self.timeout}s) en attente du résultat")
        return None

    def _strategy_to_command(self, strategy: dict) -> dict:
        """Convertit une stratégie en commande JSON pour le Bridge EA."""
        rm = strategy.get("risk_management", {})

        indicators = []
        for ind in strategy.get("indicators", []):
            ind_cmd = {
                "type": ind.get("type", ""),
                **ind.get("params", {})
            }
            indicators.append(ind_cmd)

        return {
            "strategy_id": strategy.get("id", f"strat_{int(time.time())}"),
            "symbol": strategy.get("symbol", "EURUSD"),
            "timeframe": strategy.get("timeframe", "H1"),
            "stop_loss": rm.get("stop_loss", 50),
            "take_profit": rm.get("take_profit", 100),
            "risk_percent": rm.get("risk_per_trade", 1.0),
            "entry_type": strategy.get("entry_type", "crossover"),
            "exit_type": strategy.get("exit_type", "opposite_signal"),
            "indicators": indicators,
        }

    def _parse_result(self, raw: dict) -> Dict[str, Any]:
        """Convertit le résultat du Bridge EA au format standard."""
        if raw.get("status") == "error":
            return {
                "error": raw.get("error", "Erreur inconnue"),
                "trades": 0, "profit": 0, "winrate": 0,
            }

        return {
            "trades": raw.get("trades", 0),
            "wins": raw.get("wins", 0),
            "losses": raw.get("losses", 0),
            "profit": raw.get("profit_pips", 0),
            "profit_money": raw.get("profit_money", 0),
            "starting_capital": raw.get("starting_capital", 10000),
            "ending_capital": raw.get("ending_capital", 10000),
            "winrate": raw.get("winrate", 0),
            "drawdown": raw.get("drawdown", 0),
            "drawdown_pct": raw.get("drawdown_pct", 0),
            "profit_factor": raw.get("profit_factor", 0),
            "sharpe_ratio": raw.get("sharpe_ratio", 0),
            "avg_win": raw.get("avg_win", 0),
            "avg_loss": raw.get("avg_loss", 0),
            "spread_pips": raw.get("spread_pips", 0),
            "date_start": raw.get("date_start", ""),
            "date_end": raw.get("date_end", ""),
            "bars_count": raw.get("bars_count", 0),
            "data_source": "mt5_bridge",
            "symbol": raw.get("symbol", ""),
            "trade_results": [],
            "equity_curve": [],
        }

    def _read_status(self) -> Optional[dict]:
        """Lit le fichier de statut du Bridge EA."""
        if not self.bridge_folder:
            return None
        status_file = os.path.join(self.bridge_folder, "status.json")
        if not os.path.exists(status_file):
            return None
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None