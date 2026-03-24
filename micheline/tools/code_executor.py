"""
Code Executor Tool — Exécute du Python dans un sandbox sécurisé.
Emplacement : micheline/tools/code_executor.py
"""

import sys
import io
import traceback
import threading
import ast
import copy
from typing import Any, Dict, Optional


class CodeSandbox:
    """Sandbox d'exécution Python avec restrictions de sécurité."""
    
    # Modules autorisés dans le sandbox
    ALLOWED_MODULES = {
        'math', 'statistics', 'random', 'datetime', 'time',
        'json', 'csv', 're', 'collections', 'itertools',
        'functools', 'operator', 'string', 'textwrap',
        'decimal', 'fractions', 'copy', 'pprint',
        'hashlib', 'base64', 'urllib.parse',
        'numpy', 'pandas',  # Si installés
    }
    
    # Fonctions/attributs interdits
    BLOCKED_NAMES = {
        'exec', 'eval', 'compile', '__import__',
        'open', 'file', 'input',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr',
        'breakpoint', 'exit', 'quit',
        '__builtins__', '__loader__', '__spec__',
    }
    
    # Modules explicitement interdits
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'socket', 'http', 'ftplib', 'smtplib',
        'ctypes', 'multiprocessing', 'signal',
        'importlib', 'pkgutil', 'code', 'codeop',
    }
    
    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        self.timeout = timeout
        self.max_output_size = max_output_size
    
    def _validate_code(self, code: str) -> Optional[str]:
        """Analyse statique du code avant exécution. Retourne erreur ou None."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Erreur de syntaxe : {e}"
        
        for node in ast.walk(tree):
            # Vérifier les imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split('.')[0]
                    if module_root in self.BLOCKED_MODULES:
                        return f"Module interdit : {alias.name}"
                    if module_root not in self.ALLOWED_MODULES:
                        return f"Module non autorisé : {alias.name}. Modules permis : {', '.join(sorted(self.ALLOWED_MODULES))}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split('.')[0]
                    if module_root in self.BLOCKED_MODULES:
                        return f"Module interdit : {node.module}"
                    if module_root not in self.ALLOWED_MODULES:
                        return f"Module non autorisé : {node.module}. Modules permis : {', '.join(sorted(self.ALLOWED_MODULES))}"
            
            # Vérifier les noms interdits
            elif isinstance(node, ast.Name):
                if node.id in self.BLOCKED_NAMES:
                    return f"Fonction/variable interdite : {node.id}"
            
            # Vérifier les attributs dangereux
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    if node.attr not in ('__init__', '__str__', '__repr__', '__len__',
                                          '__add__', '__sub__', '__mul__', '__eq__',
                                          '__lt__', '__gt__', '__contains__', '__iter__',
                                          '__next__', '__getitem__', '__setitem__'):
                        return f"Attribut dunder interdit : {node.attr}"
        
        return None
    
    def _create_safe_builtins(self) -> dict:
        """Crée un ensemble restreint de builtins."""
        import builtins
        
        safe = {}
        allowed_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'complex', 'dict', 'divmod', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'hasattr', 'hash',
            'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
            'len', 'list', 'map', 'max', 'min', 'next', 'object',
            'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
            'round', 'set', 'slice', 'sorted', 'str', 'sum', 'super',
            'tuple', 'type', 'zip',
            'True', 'False', 'None',
            'ArithmeticError', 'AssertionError', 'AttributeError',
            'Exception', 'IndexError', 'KeyError', 'NameError',
            'RuntimeError', 'StopIteration', 'TypeError', 'ValueError',
            'ZeroDivisionError',
        }
        
        for name in allowed_builtins:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)
        
        # __import__ restreint
        def safe_import(name, *args, **kwargs):
            module_root = name.split('.')[0]
            if module_root in self.BLOCKED_MODULES:
                raise ImportError(f"Module interdit : {name}")
            if module_root not in self.ALLOWED_MODULES:
                raise ImportError(f"Module non autorisé : {name}")
            return __builtins__['__import__'](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)
        
        safe['__import__'] = safe_import
        
        return safe
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Exécute du code Python dans le sandbox.
        
        Returns:
            {
                "success": bool,
                "output": str,        # stdout capturé
                "error": str | None,  # erreur si échec
                "result": Any,        # dernière expression évaluée
                "variables": dict     # variables créées
            }
        """
        # Validation statique
        validation_error = self._validate_code(code)
        if validation_error:
            return {
                "success": False,
                "output": "",
                "error": f"Code rejeté par le validateur : {validation_error}",
                "result": None,
                "variables": {}
            }
        
        # Préparer l'environnement sandboxé
        safe_globals = {
            '__builtins__': self._create_safe_builtins(),
            '__name__': '__sandbox__',
        }
        safe_locals = {}
        
        # Capturer stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        captured_error = io.StringIO()
        
        result_container = {"success": False, "error": None, "result": None}
        
        def run_code():
            try:
                # Essayer d'abord comme expression (pour retourner un résultat)
                try:
                    tree = ast.parse(code, mode='eval')
                    compiled = compile(tree, '<sandbox>', 'eval')
                    result_container["result"] = eval(compiled, safe_globals, safe_locals)
                    result_container["success"] = True
                except SyntaxError:
                    # Sinon exécuter comme statements
                    compiled = compile(code, '<sandbox>', 'exec')
                    exec(compiled, safe_globals, safe_locals)
                    result_container["success"] = True
            except Exception as e:
                result_container["error"] = f"{type(e).__name__}: {e}"
                result_container["success"] = False
        
        # Exécuter avec timeout
        sys.stdout = captured_output
        sys.stderr = captured_error
        
        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        if thread.is_alive():
            return {
                "success": False,
                "output": captured_output.getvalue()[:self.max_output_size],
                "error": f"Timeout : le code a dépassé {self.timeout} secondes",
                "result": None,
                "variables": {}
            }
        
        # Extraire les variables créées (filtrer les non-sérialisables)
        user_variables = {}
        for key, value in safe_locals.items():
            if not key.startswith('_'):
                try:
                    # Tester si c'est sérialisable en str au moins
                    str(value)
                    user_variables[key] = repr(value) if not isinstance(value, (int, float, str, bool, list, dict, tuple)) else value
                except Exception:
                    user_variables[key] = "<non-affichable>"
        
        output = captured_output.getvalue()
        if len(output) > self.max_output_size:
            output = output[:self.max_output_size] + "\n... [sortie tronquée]"
        
        return {
            "success": result_container["success"],
            "output": output,
            "error": result_container["error"],
            "result": repr(result_container["result"]) if result_container["result"] is not None else None,
            "variables": user_variables
        }


# Instance globale du sandbox
_sandbox = CodeSandbox()


def execute_code(code: str, timeout: int = 30) -> str:
    """
    Point d'entrée pour le tool registry.
    
    Args:
        code: Code Python à exécuter
        timeout: Timeout en secondes (défaut: 30)
    
    Returns:
        Résultat formaté en texte
    """
    if not code or not code.strip():
        return "Erreur : aucun code fourni."
    
    _sandbox.timeout = timeout
    result = _sandbox.execute(code.strip())
    
    # Formater la réponse
    parts = []
    
    if result["success"]:
        parts.append("✅ Code exécuté avec succès")
    else:
        parts.append("❌ Erreur d'exécution")
    
    if result["output"]:
        parts.append(f"\n📤 Sortie :\n{result['output']}")
    
    if result["result"] is not None:
        parts.append(f"\n🔢 Résultat : {result['result']}")
    
    if result["error"]:
        parts.append(f"\n⚠️ Erreur : {result['error']}")
    
    if result["variables"]:
        vars_str = "\n".join(f"  {k} = {v}" for k, v in result["variables"].items())
        parts.append(f"\n📦 Variables créées :\n{vars_str}")
    
    return "\n".join(parts)


# Métadonnées pour le registry
TOOL_NAME = "code_executor"
TOOL_DESCRIPTION = (
    "Exécute du code Python dans un environnement sandbox sécurisé. "
    "Peut faire des calculs, manipuler des données, créer des algorithmes. "
    "Modules autorisés : math, statistics, random, datetime, json, csv, re, "
    "collections, itertools, numpy, pandas. "
    "NE PEUT PAS accéder aux fichiers, au réseau ou au système."
)
TOOL_PARAMETERS = {
    "code": "str — Code Python à exécuter",
    "timeout": "int — Timeout en secondes (optionnel, défaut: 30)"
}