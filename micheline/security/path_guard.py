"""
PathGuard — Système de sécurité pour contrôler les accès fichiers.
Emplacement : micheline/security/path_guard.py
FICHIER MODIFIÉ — Phase 5 + Phase 6

⚠️ CE FICHIER NE DOIT JAMAIS ÊTRE MODIFIÉ PAR L'IA.
⚠️ LE FICHIER allowed_paths.json NE DOIT JAMAIS ÊTRE ACCESSIBLE À L'IA.
"""

import os
import json
from pathlib import Path
from typing import List


# Chemin vers le fichier de configuration (invisible pour l'IA)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'allowed_paths.json')

# Cache des chemins autorisés
_allowed_paths: List[str] = []
_loaded = False


def _load_allowed_paths():
    """Charge les chemins autorisés depuis le fichier JSON."""
    global _allowed_paths, _loaded

    try:
        config_path = os.path.normpath(_CONFIG_PATH)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                _allowed_paths = [
                    os.path.normpath(p) for p in config.get('allowed_paths', [])
                ]
        else:
            # Créer un fichier par défaut avec le dossier du projet
            project_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..')
            )
            _allowed_paths = [project_root]

            # Créer le dossier config si nécessaire
            config_dir = os.path.dirname(config_path)
            os.makedirs(config_dir, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'allowed_paths': [project_root],
                    '_comment': 'Ce fichier est géré par le système. NE PAS donner accès à l\'IA.'
                }, f, indent=2, ensure_ascii=False)

        _loaded = True
    except Exception as e:
        print(f"[PathGuard] Erreur de chargement : {e}")
        _allowed_paths = []
        _loaded = True


def is_allowed(path: str) -> bool:
    """
    Vérifie si un chemin est dans les racines autorisées.

    Args:
        path: Chemin à vérifier

    Returns:
        True si le chemin est autorisé, False sinon
    """
    global _loaded
    if not _loaded:
        _load_allowed_paths()

    try:
        # Résoudre le chemin absolu
        resolved = os.path.normpath(os.path.abspath(path))

        # Vérifier contre chaque racine autorisée
        for allowed_root in _allowed_paths:
            allowed_resolved = os.path.normpath(os.path.abspath(allowed_root))

            # Le chemin doit être sous (ou être) une racine autorisée
            if resolved.startswith(allowed_resolved):
                return True

            # Vérifier aussi avec os.path.commonpath
            try:
                common = os.path.commonpath([resolved, allowed_resolved])
                if common == allowed_resolved:
                    return True
            except ValueError:
                # Lecteurs différents sur Windows
                continue

        return False

    except Exception:
        return False


def validate_read(path: str) -> bool:
    """
    Valide qu'un chemin est autorisé en lecture.
    Utilisé par file_tools.py.

    Args:
        path: chemin à valider

    Returns:
        True si autorisé

    Raises:
        PermissionError si non autorisé
    """
    if not is_allowed(path):
        raise PermissionError(f"Accès lecture refusé : {path} n'est pas dans les chemins autorisés")
    return True


def validate_write(path: str) -> bool:
    """
    Valide qu'un chemin est autorisé en écriture.
    Utilisé par file_tools.py.

    Args:
        path: chemin à valider

    Returns:
        True si autorisé

    Raises:
        PermissionError si non autorisé
    """
    if not is_allowed(path):
        raise PermissionError(f"Écriture refusée : {path} n'est pas dans les chemins autorisés")
    return True


def get_guard():
    """
    Retourne un objet guard avec les méthodes de validation.
    Utilisé par file_tools.py pour accéder aux fonctions de sécurité.

    Returns:
        objet avec les méthodes is_allowed, validate_read, validate_write
    """

    class _Guard:
        """Objet guard encapsulant les fonctions de sécurité."""

        @staticmethod
        def is_allowed(path: str) -> bool:
            return is_allowed(path)

        @staticmethod
        def validate_read(path: str) -> bool:
            return validate_read(path)

        @staticmethod
        def validate_write(path: str) -> bool:
            return validate_write(path)

        @staticmethod
        def get_allowed_paths() -> List[str]:
            return get_allowed_paths_display()

    return _Guard()


def get_allowed_paths_display() -> List[str]:
    """
    Retourne les chemins autorisés pour affichage.
    Note : retourne les chemins mais PAS le contenu du fichier JSON brut.
    """
    global _loaded
    if not _loaded:
        _load_allowed_paths()

    return list(_allowed_paths)


def reload():
    """Force le rechargement des chemins autorisés."""
    global _loaded
    _loaded = False
    _load_allowed_paths()