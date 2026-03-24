"""
MetaTrader 5 Tool — Interface avec MT5 pour le trading.
Emplacement : micheline/tools/mt5_tool.py
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime


class MT5Interface:
    """Interface avec MetaTrader 5."""
    
    def __init__(self):
        self.mt5 = None
        self.connected = False
        self._try_import()
    
    def _try_import(self):
        """Tente d'importer MetaTrader5."""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            self.mt5 = None
    
    def connect(self) -> Dict[str, Any]:
        """Se connecte à MT5."""
        if self.mt5 is None:
            return {"success": False, "error": "MetaTrader5 n'est pas installé. Installe-le avec : pip install MetaTrader5"}
        
        try:
            if not self.mt5.initialize():
                return {"success": False, "error": f"Impossible d'initialiser MT5 : {self.mt5.last_error()}"}
            
            self.connected = True
            info = self.mt5.terminal_info()
            account = self.mt5.account_info()
            
            return {
                "success": True,
                "terminal": {
                    "company": info.company if info else "N/A",
                    "connected": info.connected if info else False,
                    "build": info.build if info else "N/A"
                },
                "account": {
                    "login": account.login if account else "N/A",
                    "balance": account.balance if account else 0,
                    "equity": account.equity if account else 0,
                    "currency": account.currency if account else "N/A",
                    "server": account.server if account else "N/A"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def disconnect(self):
        """Déconnecte MT5."""
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Obtient les infos d'un symbole."""
        if not self._ensure_connected():
            return {"error": "Non connecté à MT5"}
        
        try:
            info = self.mt5.symbol_info(symbol)
            if info is None:
                return {"error": f"Symbole '{symbol}' non trouvé"}
            
            tick = self.mt5.symbol_info_tick(symbol)
            
            return {
                "symbol": symbol,
                "bid": tick.bid if tick else None,
                "ask": tick.ask if tick else None,
                "spread": info.spread,
                "digits": info.digits,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "trade_mode": info.trade_mode,
                "description": info.description
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_historical_data(self, symbol: str, timeframe: str = "H1", 
                            bars: int = 100) -> Dict[str, Any]:
        """Récupère les données historiques."""
        if not self._ensure_connected():
            return {"error": "Non connecté à MT5"}
        
        try:
            # Mapper les timeframes
            tf_map = {
                "M1": self.mt5.TIMEFRAME_M1,
                "M5": self.mt5.TIMEFRAME_M5,
                "M15": self.mt5.TIMEFRAME_M15,
                "M30": self.mt5.TIMEFRAME_M30,
                "H1": self.mt5.TIMEFRAME_H1,
                "H4": self.mt5.TIMEFRAME_H4,
                "D1": self.mt5.TIMEFRAME_D1,
                "W1": self.mt5.TIMEFRAME_W1,
                "MN1": self.mt5.TIMEFRAME_MN1,
            }
            
            tf = tf_map.get(timeframe.upper())
            if tf is None:
                return {"error": f"Timeframe invalide : {timeframe}. Valides : {', '.join(tf_map.keys())}"}
            
            rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None or len(rates) == 0:
                return {"error": f"Aucune donnée pour {symbol} {timeframe}"}
            
            # Convertir en format lisible
            data = []
            for rate in rates[-10:]:  # Dernières 10 bougies pour l'affichage
                data.append({
                    "time": datetime.fromtimestamp(rate[0]).strftime('%Y-%m-%d %H:%M'),
                    "open": rate[1],
                    "high": rate[2],
                    "low": rate[3],
                    "close": rate[4],
                    "volume": rate[5]
                })
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_bars": len(rates),
                "last_10_bars": data,
                "stats": {
                    "highest_high": max(r[2] for r in rates),
                    "lowest_low": min(r[3] for r in rates),
                    "avg_volume": sum(r[5] for r in rates) / len(rates),
                    "latest_close": rates[-1][4],
                    "period_change_pct": round((rates[-1][4] - rates[0][1]) / rates[0][1] * 100, 4)
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        """Obtient les positions ouvertes."""
        if not self._ensure_connected():
            return {"error": "Non connecté à MT5"}
        
        try:
            positions = self.mt5.positions_get()
            if positions is None:
                return {"positions": [], "count": 0}
            
            pos_list = []
            for pos in positions:
                pos_list.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "comment": pos.comment
                })
            
            total_profit = sum(p["profit"] for p in pos_list)
            
            return {
                "positions": pos_list,
                "count": len(pos_list),
                "total_profit": round(total_profit, 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Obtient les infos du compte."""
        if not self._ensure_connected():
            return {"error": "Non connecté à MT5"}
        
        try:
            account = self.mt5.account_info()
            if account is None:
                return {"error": "Impossible d'obtenir les infos du compte"}
            
            return {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "free_margin": account.margin_free,
                "margin_level": account.margin_level,
                "profit": account.profit,
                "currency": account.currency,
                "leverage": account.leverage,
                "server": account.server,
                "trade_allowed": account.trade_allowed
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _ensure_connected(self) -> bool:
        """Vérifie et rétablit la connexion si nécessaire."""
        if not self.mt5:
            return False
        if not self.connected:
            result = self.connect()
            return result.get("success", False)
        return True


# Instance globale
_mt5 = MT5Interface()


def mt5_tool(action: str, symbol: str = None, timeframe: str = "H1", 
             bars: int = 100, **kwargs) -> str:
    """
    Point d'entrée pour le tool registry.
    
    Args:
        action: "connect", "disconnect", "symbol_info", "historical_data", 
                "positions", "account_info"
        symbol: Symbole (EURUSD, BTCUSD, etc.)
        timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1
        bars: Nombre de bougies historiques
    
    Returns:
        Résultat formaté en texte
    """
    action = (action or "").strip().lower()
    
    actions_map = {
        "connect": lambda: _mt5.connect(),
        "disconnect": lambda: (_mt5.disconnect(), {"success": True, "message": "Déconnecté de MT5"})[1],
        "symbol_info": lambda: _mt5.get_symbol_info(symbol) if symbol else {"error": "Symbole requis"},
        "historical_data": lambda: _mt5.get_historical_data(symbol, timeframe, bars) if symbol else {"error": "Symbole requis"},
        "history": lambda: _mt5.get_historical_data(symbol, timeframe, bars) if symbol else {"error": "Symbole requis"},
        "positions": lambda: _mt5.get_positions(),
        "account_info": lambda: _mt5.get_account_info(),
        "account": lambda: _mt5.get_account_info(),
    }
    
    if action not in actions_map:
        return f"Action MT5 inconnue : '{action}'. Actions disponibles : {', '.join(sorted(set(actions_map.keys())))}"
    
    result = actions_map[action]()
    
    # Formater le résultat
    if isinstance(result, dict):
        if "error" in result:
            return f"❌ MT5 Erreur : {result['error']}"
        
        return f"📊 MT5 — {action} :\n{json.dumps(result, indent=2, ensure_ascii=False, default=str)}"
    
    return str(result)


# Métadonnées pour le registry
TOOL_NAME = "mt5_tool"
TOOL_DESCRIPTION = (
    "Interface MetaTrader 5 pour le trading. "
    "Actions : 'connect' (connexion), 'account_info' (infos compte), "
    "'symbol_info' (infos symbole), 'historical_data' (données historiques), "
    "'positions' (positions ouvertes), 'disconnect' (déconnexion). "
    "Nécessite MetaTrader 5 installé et ouvert."
)
TOOL_PARAMETERS = {
    "action": "str — 'connect', 'account_info', 'symbol_info', 'historical_data', 'positions', 'disconnect'",
    "symbol": "str — Symbole trading (ex: 'EURUSD', 'BTCUSD') — requis pour symbol_info et historical_data",
    "timeframe": "str — M1, M5, M15, M30, H1, H4, D1, W1, MN1 (défaut: H1)",
    "bars": "int — Nombre de bougies historiques (défaut: 100)"
}