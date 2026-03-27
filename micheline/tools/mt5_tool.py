"""
micheline/tools/mt5_tools.py

Outils MT5 directs pour l'agent Micheline.
AUCUNE simulation. Connexion réelle à MetaTrader 5 uniquement.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("micheline.tools.mt5")

MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.error("❌ MetaTrader5 non installé")


def mt5_connect(params: Optional[dict] = None) -> dict:
    """
    Connecte à MetaTrader 5.
    """
    if not MT5_AVAILABLE:
        return {
            "success": False,
            "error": "MetaTrader5 non installé. pip install MetaTrader5",
            "formatted": "❌ MetaTrader5 non installé.\n\nInstallez-le avec: pip install MetaTrader5"
        }

    try:
        if mt5.initialize():
            info = mt5.terminal_info()
            account = mt5.account_info()
            result = {
                "success": True,
                "terminal": {
                    "company": info.company if info else "?",
                    "name": info.name if info else "?",
                    "build": info.build if info else 0,
                    "connected": info.connected if info else False,
                },
                "account": {
                    "login": account.login if account else 0,
                    "server": account.server if account else "?",
                    "balance": account.balance if account else 0,
                    "currency": account.currency if account else "?",
                    "leverage": account.leverage if account else 0,
                } if account else None,
            }
            result["formatted"] = (
                f"✅ **MT5 Connecté**\n"
                f"Terminal: {result['terminal']['company']} ({result['terminal']['name']})\n"
                f"Build: {result['terminal']['build']}\n"
            )
            if account:
                result["formatted"] += (
                    f"Compte: {account.login} @ {account.server}\n"
                    f"Balance: {account.balance} {account.currency}\n"
                    f"Levier: 1:{account.leverage}"
                )
            return result
        else:
            error = mt5.last_error()
            return {
                "success": False,
                "error": f"MT5 initialize échoué: {error}",
                "formatted": f"❌ Impossible de se connecter à MT5\nErreur: {error}\n\n"
                             f"Vérifiez que MetaTrader 5 est ouvert."
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "formatted": f"❌ Erreur MT5: {e}"
        }


def mt5_status(params: Optional[dict] = None) -> dict:
    """
    Retourne le statut de la connexion MT5.
    """
    if not MT5_AVAILABLE:
        return {
            "success": True,
            "connected": False,
            "mt5_installed": False,
            "formatted": "❌ MetaTrader5 non installé"
        }

    try:
        info = mt5.terminal_info()
        if info is None:
            # Pas initialisé
            return {
                "success": True,
                "connected": False,
                "mt5_installed": True,
                "initialized": False,
                "formatted": "⚠️ MT5 installé mais non initialisé. Utilisez mt5_connect."
            }

        account = mt5.account_info()
        result = {
            "success": True,
            "connected": info.connected,
            "mt5_installed": True,
            "initialized": True,
            "terminal": {
                "company": info.company,
                "name": info.name,
                "build": info.build,
                "connected": info.connected,
                "trade_allowed": info.trade_allowed,
            },
        }

        if info.connected:
            result["formatted"] = (
                f"✅ **MT5 Connecté**\n"
                f"Terminal: {info.company} ({info.name})\n"
                f"Trading autorisé: {'✅' if info.trade_allowed else '❌'}"
            )
            if account:
                result["account"] = {
                    "login": account.login,
                    "server": account.server,
                    "balance": account.balance,
                    "equity": account.equity,
                    "currency": account.currency,
                }
                result["formatted"] += (
                    f"\nCompte: {account.login} @ {account.server}\n"
                    f"Balance: {account.balance:.2f} {account.currency}\n"
                    f"Equity: {account.equity:.2f} {account.currency}"
                )
        else:
            result["formatted"] = "⚠️ MT5 initialisé mais non connecté à un serveur."

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "formatted": f"❌ Erreur MT5: {e}"
        }


def mt5_symbols(params: Optional[dict] = None) -> dict:
    """
    Liste les symboles disponibles dans MT5.
    """
    if not MT5_AVAILABLE:
        return {"success": False, "error": "MT5 non installé"}

    try:
        info = mt5.terminal_info()
        if info is None:
            mt5.initialize()

        symbols = mt5.symbols_get()
        if symbols is None:
            return {"success": False, "error": "Impossible de récupérer les symboles"}

        # Filtrer les symboles visibles
        group = (params or {}).get("group", "")
        if group:
            symbols = [s for s in symbols if group.upper() in s.name.upper()]

        symbol_list = []
        for s in symbols[:100]:  # Limiter à 100
            if s.visible:
                symbol_list.append({
                    "name": s.name,
                    "description": s.description,
                    "spread": s.spread,
                    "digits": s.digits,
                    "trade_mode": s.trade_mode,
                })

        lines = [f"📋 **Symboles MT5** ({len(symbol_list)} visibles)"]
        lines.append("━" * 40)
        for s in symbol_list[:30]:
            lines.append(f"  • {s['name']} ({s['description']}) | Spread: {s['spread']} | Digits: {s['digits']}")
        if len(symbol_list) > 30:
            lines.append(f"  ... et {len(symbol_list) - 30} de plus")

        return {
            "success": True,
            "count": len(symbol_list),
            "symbols": symbol_list,
            "formatted": "\n".join(lines)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def mt5_disconnect(params: Optional[dict] = None) -> dict:
    """Déconnecte MT5."""
    if not MT5_AVAILABLE:
        return {"success": True, "formatted": "MT5 non installé, rien à déconnecter."}

    try:
        mt5.shutdown()
        return {
            "success": True,
            "formatted": "✅ MT5 déconnecté."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}