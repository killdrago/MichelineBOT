"""
Path Guard — Système de sécurité pour les accès fichiers.

RÈGLES ABSOLUES:
1. L'IA ne peut JAMAIS accéder directement au système de fichiers
2. Tous les accès passent par ce module
3. Ce module est NON MODIFIABLE par l'IA
4. Le fichier allowed_paths.json est INVISIBLE pour l'IA
5. L'IA ne peut JAMAIS contourner ces règles

Ce fichier est chargé par le SYSTÈME uniquement.
"""

import os
import json
from typing import Optional


# Chemin du fichier de configuration (relatif au module security)
_SECURITY_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SECURITY_DIR, "allowed_paths.json")


class PathGuard:
    """
    Contrôle tous les accès fichiers.
    Chargé une seule fois au démarrage.
    L'IA n'a AUCUN accès à cette classe ni à sa configuration.
    """

    def __init__(self):
        self._config = self._load_config()
        self._allowed_paths = []
        self._allowed_extensions_read = []
        self._allowed_extensions_write = []
        self._blocked_patterns = []
        self._max_file_size_bytes = 50 * 1024 * 1024  # 50 MB par défaut

        self._parse_config()
        print(f"[PathGuard] ✅ Initialisé | {len(self._allowed_paths)} chemin(s) autorisé(s)")

    def _load_config(self) -> dict:
        """Charge le fichier de configuration. SYSTÈME UNIQUEMENT."""
        try:
            if not os.path.isfile(_CONFIG_PATH):
                print(f"[PathGuard] ⚠️ Fichier config introuvable: {_CONFIG_PATH}")
                print(f"[PathGuard] ⚠️ Aucun accès fichier ne sera autorisé.")
                return {}

            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data

        except Exception as e:
            print(f"[PathGuard] ❌ Erreur chargement config: {e}")
            return {}

    def _parse_config(self):
        """Parse la configuration chargée."""
        # Chemins autorisés (normalisés)
        raw_paths = self._config.get("allowed_paths", [])
        self._allowed_paths = []
        for p in raw_paths:
            try:
                normalized = os.path.abspath(os.path.normpath(os.path.expanduser(p)))
                if os.path.isdir(normalized):
                    self._allowed_paths.append(normalized)
                else:
                    print(f"[PathGuard] ⚠️ Chemin autorisé introuvable (ignoré): {p}")
            except Exception:
                pass

        # Extensions autorisées
        self._allowed_extensions_read = [
            ext.lower() for ext in self._config.get("allowed_extensions_read", [])
        ]
        self._allowed_extensions_write = [
            ext.lower() for ext in self._config.get("allowed_extensions_write", [])
        ]

        # Patterns bloqués
        self._blocked_patterns = [
            pat.lower() for pat in self._config.get("blocked_patterns", [])
        ]

        # Taille max
        max_mb = self._config.get("max_file_size_mb", 50)
        try:
            self._max_file_size_bytes = int(float(max_mb) * 1024 * 1024)
        except Exception:
            self._max_file_size_bytes = 50 * 1024 * 1024

    def _normalize_path(self, path: str) -> str:
        """Normalise un chemin pour comparaison sécurisée."""
        try:
            return os.path.abspath(os.path.normpath(os.path.expanduser(path.strip())))
        except Exception:
            return ""

    def _is_under_allowed_root(self, normalized_path: str) -> bool:
        """Vérifie si le chemin est sous une racine autorisée."""
        if not normalized_path:
            return False

        for root in self._allowed_paths:
            try:
                # Vérifie que le chemin commence bien par la racine
                common = os.path.commonpath([root, normalized_path])
                if common == root:
                    return True
            except (ValueError, TypeError):
                continue

        return False

    def _has_blocked_pattern(self, normalized_path: str) -> Optional[str]:
        """Vérifie si le chemin contient un pattern bloqué. Retourne le pattern ou None."""
        path_lower = normalized_path.lower()

        for pattern in self._blocked_patterns:
            if pattern in path_lower:
                return pattern

        return None

    def _check_extension(self, normalized_path: str, mode: str = "read") -> bool:
        """Vérifie si l'extension du fichier est autorisée."""
        _, ext = os.path.splitext(normalized_path)
        ext = ext.lower()

        if not ext:
            return True  # Pas d'extension = dossier ou fichier sans ext

        if mode == "write":
            if not self._allowed_extensions_write:
                return True  # Pas de restriction
            return ext in self._allowed_extensions_write
        else:
            if not self._allowed_extensions_read:
                return True
            return ext in self._allowed_extensions_read

    def is_allowed(self, path: str, mode: str = "read") -> bool:
        """
        Vérifie si un accès fichier est autorisé.

        Args:
            path: le chemin à vérifier
            mode: "read" ou "write"

        Returns:
            True si l'accès est autorisé, False sinon
        """
        if not path or not path.strip():
            return False

        normalized = self._normalize_path(path)
        if not normalized:
            return False

        # 1. Vérifier les patterns bloqués
        blocked = self._has_blocked_pattern(normalized)
        if blocked:
            print(f"[PathGuard] 🔒 BLOQUÉ (pattern interdit '{blocked}'): {path}")
            return False

        # 2. Vérifier si sous une racine autorisée
        if not self._is_under_allowed_root(normalized):
            print(f"[PathGuard] 🔒 BLOQUÉ (hors racines autorisées): {path}")
            return False

        # 3. Vérifier l'extension
        if not self._check_extension(normalized, mode):
            print(f"[PathGuard] 🔒 BLOQUÉ (extension non autorisée en {mode}): {path}")
            return False

        return True

    def check_file_size(self, path: str) -> bool:
        """Vérifie que le fichier ne dépasse pas la taille maximale."""
        try:
            normalized = self._normalize_path(path)
            if os.path.isfile(normalized):
                size = os.path.getsize(normalized)
                if size > self._max_file_size_bytes:
                    max_mb = self._max_file_size_bytes / (1024 * 1024)
                    actual_mb = size / (1024 * 1024)
                    print(f"[PathGuard] 🔒 BLOQUÉ (taille {actual_mb:.1f}MB > max {max_mb:.1f}MB): {path}")
                    return False
            return True
        except Exception:
            return True

    def validate_read(self, path: str) -> tuple:
        """
        Valide un accès en lecture.

        Returns:
            (allowed: bool, normalized_path: str, error: str or None)
        """
        normalized = self._normalize_path(path)

        if not self.is_allowed(path, mode="read"):
            return (False, normalized, f"Accès lecture refusé: {path}")

        if not os.path.exists(normalized):
            return (False, normalized, f"Fichier introuvable: {path}")

        if not self.check_file_size(normalized):
            return (False, normalized, f"Fichier trop volumineux: {path}")

        return (True, normalized, None)

    def validate_write(self, path: str) -> tuple:
        """
        Valide un accès en écriture.

        Returns:
            (allowed: bool, normalized_path: str, error: str or None)
        """
        normalized = self._normalize_path(path)

        if not self.is_allowed(path, mode="write"):
            return (False, normalized, f"Accès écriture refusé: {path}")

        # Vérifier que le dossier parent existe
        parent = os.path.dirname(normalized)
        if not os.path.isdir(parent):
            return (False, normalized, f"Dossier parent introuvable: {parent}")

        return (True, normalized, None)

    def get_allowed_roots(self) -> list:
        """Retourne la liste des racines autorisées (pour information, PAS le fichier config)."""
        return list(self._allowed_paths)


# === INSTANCE GLOBALE (singleton) ===
# Chargée une seule fois au démarrage du système.
# L'IA n'a PAS accès à cette instance directement.

_guard_instance = None


def get_guard() -> PathGuard:
    """Retourne l'instance unique du PathGuard."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = PathGuard()
    return _guard_instance


def is_allowed(path: str, mode: str = "read") -> bool:
    """Raccourci pour vérifier un accès."""
    return get_guard().is_allowed(path, mode)


def validate_read(path: str) -> tuple:
    """Raccourci pour valider un accès en lecture."""
    return get_guard().validate_read(path)


def validate_write(path: str) -> tuple:
    """Raccourci pour valider un accès en écriture."""
    return get_guard().validate_write(path)