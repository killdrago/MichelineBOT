"""
Memory — Mémoire persistante de l'agent Micheline.
Stocke les expériences, découvertes et apprentissages.

Base SQLite séparée de la mémoire conversationnelle.
Permet à l'agent de:
- Ne jamais refaire la même erreur
- Se souvenir des stratégies testées
- Retrouver des résultats passés
- Accumuler des connaissances au fil du temps
"""

import os
import json
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any


# Base de données mémoire agent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "experiments.db")


class AgentMemory:
    """
    Mémoire persistante de l'agent.
    Stocke 3 types de données:
    1. Expériences (actions exécutées + résultats)
    2. Découvertes (faits appris, patterns détectés)
    3. Stratégies (configurations trading testées + scores)
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        print(f"[AgentMemory] ✅ Mémoire initialisée: {self.db_path}")

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            # Table des expériences (chaque action de l'agent)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    objective TEXT,
                    action TEXT NOT NULL,
                    params TEXT,
                    result TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    score REAL,
                    tags TEXT,
                    notes TEXT
                )
            """)

            # Table des découvertes (faits appris)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    UNIQUE(category, key)
                )
            """)

            # Table des stratégies trading
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    name TEXT NOT NULL,
                    symbol TEXT,
                    config TEXT NOT NULL,
                    profit REAL,
                    drawdown REAL,
                    sharpe REAL,
                    winrate REAL,
                    trades INTEGER,
                    score REAL,
                    status TEXT DEFAULT 'tested',
                    notes TEXT
                )
            """)

            # Index
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiences(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_action ON experiences(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_success ON experiences(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_disc_category ON discoveries(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_symbol ON strategies(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_score ON strategies(score DESC)")

            conn.commit()

    # ==========================================
    # EXPÉRIENCES
    # ==========================================

    def store_experience(self, objective: str, action: str, params: dict,
                         result: str, success: bool, score: float = None,
                         tags: list = None, notes: str = None):
        """Enregistre une expérience (action + résultat)."""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT INTO experiences 
                        (timestamp, objective, action, params, result, success, score, tags, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        objective or "",
                        action,
                        json.dumps(params or {}, ensure_ascii=False),
                        str(result or "")[:5000],
                        1 if success else 0,
                        score,
                        json.dumps(tags or [], ensure_ascii=False),
                        notes
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[AgentMemory] Erreur store_experience: {e}")

    def get_recent_experiences(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère les N dernières expériences."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur get_recent: {e}")
            return []

    def get_experiences_by_action(self, action: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les expériences pour une action spécifique."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM experiences WHERE action = ? ORDER BY timestamp DESC LIMIT ?",
                    (action, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur get_by_action: {e}")
            return []

    def get_failed_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les dernières expériences échouées (pour ne pas les répéter)."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM experiences WHERE success = 0 ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur get_failed: {e}")
            return []

    def search_experiences(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Recherche dans les expériences par mots-clés."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT * FROM experiences 
                    WHERE objective LIKE ? OR action LIKE ? OR result LIKE ? OR notes LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur search: {e}")
            return []

    # ==========================================
    # DÉCOUVERTES
    # ==========================================

    def store_discovery(self, category: str, key: str, value: str,
                        confidence: float = 0.5, source: str = None):
        """
        Enregistre une découverte (fait appris).
        Si la clé existe déjà, met à jour la valeur et la confiance.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT INTO discoveries (timestamp, category, key, value, confidence, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(category, key) DO UPDATE SET
                            value = excluded.value,
                            confidence = excluded.confidence,
                            source = excluded.source,
                            timestamp = excluded.timestamp
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        category,
                        key,
                        value,
                        confidence,
                        source
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[AgentMemory] Erreur store_discovery: {e}")

    def get_discoveries(self, category: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les découvertes, optionnellement filtrées par catégorie."""
        try:
            with self._get_conn() as conn:
                if category:
                    rows = conn.execute(
                        "SELECT * FROM discoveries WHERE category = ? ORDER BY confidence DESC LIMIT ?",
                        (category, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM discoveries ORDER BY timestamp DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur get_discoveries: {e}")
            return []

    def get_discovery(self, category: str, key: str) -> Optional[Dict[str, Any]]:
        """Récupère une découverte spécifique."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM discoveries WHERE category = ? AND key = ?",
                    (category, key)
                ).fetchone()
                return dict(row) if row else None
        except Exception as e:
            return None

    # ==========================================
    # STRATÉGIES TRADING
    # ==========================================

    def store_strategy(self, name: str, symbol: str, config: dict,
                       profit: float = None, drawdown: float = None,
                       sharpe: float = None, winrate: float = None,
                       trades: int = None, score: float = None,
                       status: str = "tested", notes: str = None):
        """Enregistre une stratégie trading testée."""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT INTO strategies
                        (timestamp, name, symbol, config, profit, drawdown,
                         sharpe, winrate, trades, score, status, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        name,
                        symbol or "",
                        json.dumps(config, ensure_ascii=False),
                        profit, drawdown, sharpe, winrate, trades, score,
                        status,
                        notes
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[AgentMemory] Erreur store_strategy: {e}")

    def get_best_strategies(self, symbol: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les meilleures stratégies par score."""
        try:
            with self._get_conn() as conn:
                if symbol:
                    rows = conn.execute(
                        "SELECT * FROM strategies WHERE symbol = ? AND score IS NOT NULL ORDER BY score DESC LIMIT ?",
                        (symbol, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM strategies WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[AgentMemory] Erreur get_best_strategies: {e}")
            return []

    def get_strategies_by_status(self, status: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère les stratégies par statut (tested, validated, rejected, deployed)."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM strategies WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            return []

    # ==========================================
    # RÉSUMÉ / CONTEXTE POUR LE LLM
    # ==========================================

    def get_context_summary(self, max_items: int = 5) -> str:
        """
        Génère un résumé de la mémoire pour enrichir le contexte du LLM.
        Utilisé par le Planner pour prendre de meilleures décisions.
        """
        parts = []

        # Dernières expériences
        recent = self.get_recent_experiences(limit=max_items)
        if recent:
            parts.append("=== Expériences récentes ===")
            for exp in recent:
                status = "✅" if exp.get("success") else "❌"
                parts.append(
                    f"  {status} [{exp.get('action')}] {exp.get('objective', '')[:80]} "
                    f"→ {exp.get('result', '')[:100]}"
                )

        # Échecs récents (pour ne pas les répéter)
        failures = self.get_failed_experiences(limit=3)
        if failures:
            parts.append("\n=== Erreurs à éviter ===")
            for f in failures:
                parts.append(f"  ❌ {f.get('action')}: {f.get('result', '')[:100]}")

        # Découvertes clés
        discoveries = self.get_discoveries(limit=max_items)
        if discoveries:
            parts.append("\n=== Découvertes ===")
            for d in discoveries:
                parts.append(f"  [{d.get('category')}] {d.get('key')}: {d.get('value', '')[:100]}")

        # Meilleures stratégies
        best = self.get_best_strategies(limit=3)
        if best:
            parts.append("\n=== Meilleures stratégies ===")
            for s in best:
                parts.append(
                    f"  {s.get('name')} ({s.get('symbol', '?')}) "
                    f"score={s.get('score', '?')} profit={s.get('profit', '?')}%"
                )

        if not parts:
            return "Aucune mémoire enregistrée."

        return "\n".join(parts)

    # ==========================================
    # STATS / MAINTENANCE
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la mémoire."""
        try:
            with self._get_conn() as conn:
                exp_count = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
                exp_success = conn.execute("SELECT COUNT(*) FROM experiences WHERE success = 1").fetchone()[0]
                disc_count = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
                strat_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]

                return {
                    "experiences_total": exp_count,
                    "experiences_success_rate": round(exp_success / max(exp_count, 1) * 100, 1),
                    "discoveries_total": disc_count,
                    "strategies_total": strat_count,
                    "db_path": self.db_path,
                    "db_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            return {"error": str(e)}

    def purge_old(self, days: int = 90):
        """Purge les expériences plus vieilles que N jours."""
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM experiences WHERE timestamp < datetime('now', ?)",
                    (f"-{days} days",)
                )
                deleted = cur.rowcount or 0
                conn.commit()
                if deleted:
                    print(f"[AgentMemory] Purge: {deleted} expériences supprimées (> {days} jours)")
        except Exception as e:
            print(f"[AgentMemory] Erreur purge: {e}")