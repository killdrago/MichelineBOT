"""
File Tools — Outils d'accès fichiers SÉCURISÉS.
Tous les accès passent par le PathGuard.
L'IA ne peut JAMAIS contourner ces contrôles.
"""

import os

# Import du système de sécurité
from micheline.security.path_guard import validate_read, validate_write, get_guard


def tool_read_file(params: dict) -> dict:
    """
    Lit le contenu d'un fichier texte.
    SÉCURISÉ: passe par PathGuard.
    """
    path = params.get("path", "")
    if not path:
        return {"success": False, "error": "Aucun chemin de fichier fourni."}

    # === CONTRÔLE DE SÉCURITÉ ===
    allowed, normalized, error = validate_read(path)
    if not allowed:
        return {"success": False, "error": f"🔒 {error}"}

    try:
        max_chars = int(params.get("max_chars", 50000))

        with open(normalized, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)

        truncated = len(content) >= max_chars
        file_size = os.path.getsize(normalized)

        return {
            "success": True,
            "data": {
                "path": normalized,
                "content": content,
                "size_bytes": file_size,
                "truncated": truncated,
                "filename": os.path.basename(normalized)
            }
        }

    except Exception as e:
        return {"success": False, "error": f"Erreur lecture: {str(e)}"}


def tool_write_file(params: dict) -> dict:
    """
    Écrit du contenu dans un fichier texte.
    SÉCURISÉ: passe par PathGuard.
    """
    path = params.get("path", "")
    content = params.get("content", "")
    mode = params.get("mode", "write")  # "write" ou "append"

    if not path:
        return {"success": False, "error": "Aucun chemin de fichier fourni."}
    if not content:
        return {"success": False, "error": "Aucun contenu à écrire."}

    # === CONTRÔLE DE SÉCURITÉ ===
    allowed, normalized, error = validate_write(path)
    if not allowed:
        return {"success": False, "error": f"🔒 {error}"}

    try:
        file_mode = "a" if mode == "append" else "w"

        with open(normalized, file_mode, encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "data": {
                "path": normalized,
                "mode": mode,
                "bytes_written": len(content.encode("utf-8")),
                "filename": os.path.basename(normalized)
            }
        }

    except Exception as e:
        return {"success": False, "error": f"Erreur écriture: {str(e)}"}


def tool_file_info(params: dict) -> dict:
    """
    Retourne les métadonnées d'un fichier (taille, dates, etc.).
    SÉCURISÉ: passe par PathGuard.
    """
    path = params.get("path", "")
    if not path:
        return {"success": False, "error": "Aucun chemin fourni."}

    # === CONTRÔLE DE SÉCURITÉ ===
    allowed, normalized, error = validate_read(path)
    if not allowed:
        return {"success": False, "error": f"🔒 {error}"}

    try:
        stat = os.stat(normalized)
        from datetime import datetime

        return {
            "success": True,
            "data": {
                "path": normalized,
                "filename": os.path.basename(normalized),
                "size_bytes": stat.st_size,
                "size_readable": _human_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "is_file": os.path.isfile(normalized),
                "is_dir": os.path.isdir(normalized),
                "extension": os.path.splitext(normalized)[1].lower()
            }
        }

    except Exception as e:
        return {"success": False, "error": f"Erreur info fichier: {str(e)}"}


def tool_list_allowed_roots(params: dict) -> dict:
    """
    Liste les racines autorisées (sans révéler le fichier de config).
    L'IA sait OÙ elle peut travailler, mais pas COMMENT c'est configuré.
    """
    guard = get_guard()
    roots = guard.get_allowed_roots()

    return {
        "success": True,
        "data": {
            "allowed_roots": roots,
            "count": len(roots),
            "note": "Vous pouvez lire/écrire des fichiers uniquement dans ces dossiers."
        }
    }


def _human_size(size_bytes: int) -> str:
    """Convertit des bytes en taille lisible."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def register_file_tools(registry):
    """Enregistre tous les outils fichiers sécurisés dans le registry."""

    registry.register(
        name="read_file",
        description="Lit le contenu d'un fichier texte (accès sécurisé)",
        function=tool_read_file,
        param_schema={
            "path": "string - chemin du fichier à lire",
            "max_chars": "int (optionnel, défaut 50000) - nombre max de caractères"
        },
        requires_security=True
    )

    registry.register(
        name="write_file",
        description="Écrit du contenu dans un fichier texte (accès sécurisé)",
        function=tool_write_file,
        param_schema={
            "path": "string - chemin du fichier",
            "content": "string - contenu à écrire",
            "mode": "string (optionnel) - 'write' (écrase) ou 'append' (ajoute)"
        },
        requires_security=True
    )

    registry.register(
        name="file_info",
        description="Retourne les métadonnées d'un fichier (taille, dates)",
        function=tool_file_info,
        param_schema={"path": "string - chemin du fichier"},
        requires_security=True
    )

    registry.register(
        name="list_allowed_paths",
        description="Liste les dossiers où l'IA peut lire/écrire des fichiers",
        function=tool_list_allowed_roots,
        param_schema={}
    )

    print(f"[Tools] Outils fichiers sécurisés enregistrés.")