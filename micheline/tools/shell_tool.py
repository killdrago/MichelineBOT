"""
Shell Command Tool — Exécute des commandes système contrôlées.
Emplacement : micheline/tools/shell_tool.py
"""

import subprocess
import os
import json
import shlex
from typing import Dict, Any, Optional


class ShellGuard:
    """Contrôle les commandes shell autorisées."""
    
    def __init__(self):
        self.allowed_commands = set()
        self.blocked_patterns = set()
        self._load_config()
    
    def _load_config(self):
        """Charge la whitelist de commandes depuis allowed_commands.json."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'allowed_commands.json'
        )
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.allowed_commands = set(config.get('allowed_commands', []))
                    self.blocked_patterns = set(config.get('blocked_patterns', []))
            else:
                # Défaut restrictif si pas de fichier config
                self.allowed_commands = {
                    'dir', 'echo', 'type', 'find', 'findstr',
                    'systeminfo', 'hostname', 'whoami', 'date', 'time',
                    'ping', 'ipconfig', 'tasklist', 'wmic',
                    'python', 'pip', 'git',
                    'ls', 'cat', 'grep', 'pwd', 'uname',  # Linux
                }
                self.blocked_patterns = {
                    'del ', 'rm ', 'rmdir', 'format', 'fdisk',
                    'shutdown', 'restart', 'reboot',
                    'reg ', 'regedit', 'net user', 'net localgroup',
                    'powershell', 'cmd /c', 'cmd.exe',
                    '> ', '>> ', '| del', '| rm',
                    'mklink', 'takeown', 'icacls',
                }
                # Créer le fichier par défaut
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'allowed_commands': sorted(self.allowed_commands),
                        'blocked_patterns': sorted(self.blocked_patterns),
                        '_comment': 'Ce fichier contrôle les commandes shell autorisées. NE PAS donner accès à l\'IA.'
                    }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # En cas d'erreur, mode ultra-restrictif
            self.allowed_commands = {'echo', 'dir', 'date', 'hostname'}
            self.blocked_patterns = set()
    
    def is_allowed(self, command: str) -> tuple:
        """
        Vérifie si une commande est autorisée.
        
        Returns:
            (bool, str) — (autorisé, raison si refusé)
        """
        if not command or not command.strip():
            return False, "Commande vide"
        
        command_lower = command.lower().strip()
        
        # Vérifier les patterns bloqués
        for pattern in self.blocked_patterns:
            if pattern.lower() in command_lower:
                return False, f"Pattern bloqué détecté : '{pattern}'"
        
        # Extraire le nom de la commande (premier mot)
        parts = command_lower.split()
        cmd_name = parts[0] if parts else ''
        
        # Retirer le chemin s'il y en a un
        cmd_name = os.path.basename(cmd_name)
        # Retirer l'extension .exe si présente
        if cmd_name.endswith('.exe'):
            cmd_name = cmd_name[:-4]
        
        if cmd_name not in self.allowed_commands:
            return False, f"Commande non autorisée : '{cmd_name}'. Commandes permises : {', '.join(sorted(self.allowed_commands))}"
        
        return True, "OK"


# Instance globale
_guard = ShellGuard()


def shell_command(command: str, working_directory: str = None, timeout: int = 30) -> str:
    """
    Point d'entrée pour le tool registry.
    
    Args:
        command: Commande à exécuter
        working_directory: Répertoire de travail (optionnel)
        timeout: Timeout en secondes (défaut: 30)
    
    Returns:
        Résultat formaté en texte
    """
    if not command or not command.strip():
        return "Erreur : aucune commande fournie."
    
    command = command.strip()
    
    # Vérifier si la commande est autorisée
    allowed, reason = _guard.is_allowed(command)
    if not allowed:
        return f"🚫 Commande REFUSÉE : {reason}"
    
    # Vérifier le répertoire de travail avec PathGuard
    if working_directory:
        try:
            from micheline.security.path_guard import is_allowed as path_allowed
            if not path_allowed(working_directory):
                return f"🚫 Répertoire non autorisé : {working_directory}"
        except ImportError:
            pass
    
    try:
        # Timeout court pour les commandes "start" (elles lancent et reviennent vite)
        effective_timeout = timeout
        if command.lower().strip().startswith('start'):
            effective_timeout = 5
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=effective_timeout,
            cwd=working_directory,
        )
        
        def decode_output(raw_bytes):
            """Décode la sortie en essayant plusieurs encodages."""
            if not raw_bytes:
                return ""
            for encoding in ['utf-8', 'cp850', 'cp1252', 'latin-1']:
                try:
                    return raw_bytes.decode(encoding)
                except (UnicodeDecodeError, AttributeError):
                    continue
            return raw_bytes.decode('utf-8', errors='replace')
        
        stdout = decode_output(result.stdout).strip()
        stderr = decode_output(result.stderr).strip()
        
        parts = []
        parts.append(f"💻 Commande : {command}")
        
        if result.returncode == 0:
            parts.append("✅ Succès")
        else:
            parts.append("❌ Échec")
        
        parts.append(f"📊 Code retour : {result.returncode}")
        
        if stdout:
            if len(stdout) > 5000:
                stdout = stdout[:5000] + "\n... [sortie tronquée]"
            parts.append(f"\n📤 Sortie :\n{stdout}")
        
        if stderr:
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n... [erreurs tronquées]"
            parts.append(f"\n⚠️ Erreurs :\n{stderr}")
        
        return "\n".join(parts)
        
    except subprocess.TimeoutExpired:
        return f"⏰ Timeout : la commande a dépassé {timeout} secondes."
    except FileNotFoundError:
        return f"❌ Commande introuvable : {command.split()[0]}"
    except Exception as e:
        return f"❌ Erreur : {type(e).__name__}: {e}"
        

# Métadonnées pour le registry
TOOL_NAME = "shell_command"
TOOL_DESCRIPTION = (
    "Exécute une commande système dans un terminal contrôlé. "
    "Seules les commandes whitelistées sont autorisées. "
    "Commandes typiques : dir, echo, ping, systeminfo, python, pip, git. "
    "Les commandes destructives (del, rm, format, shutdown) sont INTERDITES."
)
TOOL_PARAMETERS = {
    "command": "str — Commande à exécuter",
    "working_directory": "str — Répertoire de travail (optionnel)",
    "timeout": "int — Timeout en secondes (défaut: 30)"
}