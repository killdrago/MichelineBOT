"""
App Launcher Tool — Lance n'importe quel logiciel sur le système.
Emplacement : micheline/tools/app_launcher.py
"""

import subprocess
import os
import re
import glob
from typing import List, Dict, Optional


# Aliases courants (nom parlé → nom réel de l'exécutable)
COMMON_ALIASES = {
    # Français
    "bloc note": "notepad",
    "bloc notes": "notepad",
    "bloc-note": "notepad",
    "bloc-notes": "notepad",
    "calculatrice": "calc",
    "explorateur": "explorer",
    "explorateur de fichiers": "explorer",
    "navigateur": "start https://www.google.com",
    "gestionnaire de tâches": "taskmgr",
    "gestionnaire de taches": "taskmgr",
    "invite de commande": "cmd",
    "terminal": "cmd",
    "editeur de texte": "notepad",
    "éditeur de texte": "notepad",
    "capture d'écran": "snippingtool",
    "capture d'ecran": "snippingtool",
    "outil capture": "snippingtool",
    "informations système": "msinfo32",
    "informations systeme": "msinfo32",
    "panneau de configuration": "control",
    "panneau de config": "control",
    "paramètres": "ms-settings:",
    "parametres": "ms-settings:",
    "nettoyage de disque": "cleanmgr",
    
    # Anglais courant
    "task manager": "taskmgr",
    "file explorer": "explorer",
    "command prompt": "cmd",
    "snipping tool": "snippingtool",
    "system info": "msinfo32",
    "control panel": "control",
    "settings": "ms-settings:",
    "disk cleanup": "cleanmgr",
    
    # Logiciels populaires (noms parlés → exécutables)
    "paint": "mspaint",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "visual studio code": "code",
    "vscode": "code",
    "vs code": "code",
    "steam": "steam",
    "discord": "discord",
    "spotify": "spotify",
    "vlc": "vlc",
    "obs": "obs64",
    "obs studio": "obs64",
    "gimp": "gimp",
    "audacity": "audacity",
    "winrar": "winrar",
    "7zip": "7zFM",
    "7-zip": "7zFM",
    "notepad++": "notepad++",
    "sublime text": "sublime_text",
    "sublime": "sublime_text",
}


def _find_executable(name: str) -> Optional[str]:
    """
    Cherche un exécutable sur le système Windows.
    
    Stratégie :
    1. Vérifier les aliases connus
    2. Chercher dans PATH
    3. Chercher dans Program Files
    4. Chercher dans le menu Démarrer
    5. Utiliser le nom tel quel
    """
    name_lower = name.lower().strip()
    
    # 1. Vérifier les aliases
    if name_lower in COMMON_ALIASES:
        return COMMON_ALIASES[name_lower]
    
    # 2. Vérifier si c'est déjà un exécutable valide dans PATH
    for ext in ['', '.exe', '.bat', '.cmd', '.com']:
        try:
            result = subprocess.run(
                ['where', f'{name_lower}{ext}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0].strip()
        except Exception:
            continue
    
    # 3. Chercher dans Program Files
    search_dirs = [
        os.environ.get('ProgramFiles', 'C:\\Program Files'),
        os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs'),
        os.environ.get('APPDATA', ''),
    ]
    
    for search_dir in search_dirs:
        if not search_dir or not os.path.exists(search_dir):
            continue
        try:
            # Chercher récursivement (limité à 2 niveaux pour la vitesse)
            for root, dirs, files in os.walk(search_dir):
                depth = root.replace(search_dir, '').count(os.sep)
                if depth > 3:
                    dirs.clear()
                    continue
                for f in files:
                    if f.lower().endswith('.exe'):
                        f_name = f.lower().replace('.exe', '')
                        if name_lower in f_name or f_name in name_lower:
                            return os.path.join(root, f)
        except PermissionError:
            continue
    
    # 4. Chercher dans le menu Démarrer (raccourcis .lnk)
    start_menu_dirs = [
        os.path.join(os.environ.get('APPDATA', ''), 
                      'Microsoft', 'Windows', 'Start Menu', 'Programs'),
        os.path.join(os.environ.get('ProgramData', 'C:\\ProgramData'), 
                      'Microsoft', 'Windows', 'Start Menu', 'Programs'),
    ]
    
    for start_dir in start_menu_dirs:
        if not os.path.exists(start_dir):
            continue
        try:
            for root, dirs, files in os.walk(start_dir):
                for f in files:
                    if f.lower().endswith('.lnk'):
                        f_name = f.lower().replace('.lnk', '')
                        if name_lower in f_name or f_name in name_lower:
                            return os.path.join(root, f)
        except PermissionError:
            continue
    
    # 5. Retourner le nom tel quel (Windows essaiera de le trouver)
    return name_lower


def app_launcher(app_names: list = None, app_name: str = None) -> str:
    """
    Lance un ou plusieurs logiciels.
    
    Args:
        app_names: Liste de noms de logiciels à ouvrir
        app_name: Nom d'un seul logiciel (alternatif)
    
    Returns:
        Résultat formaté en texte
    """
    # Gérer les deux formats possibles
    if app_names is None and app_name:
        app_names = [app_name]
    if app_names is None:
        return "Erreur : aucun logiciel spécifié."
    if isinstance(app_names, str):
        app_names = [app_names]
    
    results = []
    
    for name in app_names:
        name = name.strip()
        if not name:
            continue
        
        # Chercher l'exécutable
        executable = _find_executable(name)
        
        if not executable:
            results.append(f"❌ '{name}' : logiciel introuvable sur le système")
            continue
        
        try:
            # Construire la commande
            if executable.endswith('.lnk'):
                # Raccourci Windows : ouvrir avec start
                cmd = f'start "" "{executable}"'
            elif executable.startswith('ms-'):
                # URI Windows (ms-settings:, etc.)
                cmd = f'start {executable}'
            elif os.path.isabs(executable):
                # Chemin absolu
                cmd = f'start "" "{executable}"'
            else:
                # Nom simple (dans PATH)
                cmd = f'start /B {executable}'
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # Attendre un tout petit peu pour voir si ça crash immédiatement
            import time
            time.sleep(0.5)
            
            if process.poll() is not None and process.returncode != 0:
                results.append(f"❌ '{name}' : échec du lancement (code {process.returncode})")
            else:
                results.append(f"✅ '{name}' lancé ({executable})")
                
        except FileNotFoundError:
            results.append(f"❌ '{name}' : commande '{executable}' introuvable")
        except Exception as e:
            results.append(f"❌ '{name}' : erreur — {type(e).__name__}: {e}")
    
    if not results:
        return "Aucun logiciel à lancer."
    
    header = "🚀 Lancement de logiciels :\n"
    return header + "\n".join(f"  {r}" for r in results)


# Métadonnées pour le registry
TOOL_NAME = "app_launcher"
TOOL_DESCRIPTION = (
    "Lance un ou plusieurs logiciels sur le système. "
    "Peut ouvrir n'importe quel programme installé par son nom. "
    "Exemples : 'paint', 'chrome', 'bloc note', 'excel', 'discord', 'vscode'. "
    "Cherche automatiquement dans PATH, Program Files et le menu Démarrer."
)
TOOL_PARAMETERS = {
    "app_names": "list — Liste de noms de logiciels à ouvrir",
    "app_name": "str — Nom d'un seul logiciel (alternatif)"
}