"""
Planner — Décompose un objectif en étapes actionnables.
Utilise le LLM local de Micheline.
Connaît la liste des outils disponibles pour planifier correctement.
"""

import json
from datetime import datetime


class Plan:
    def __init__(self, objective: str, steps: list):
        self.objective = objective
        self.steps = steps
        self.current_step = 0
        self.created_at = datetime.now().isoformat()
        self.status = "pending"

    def next_step(self):
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            self.status = "in_progress"
            return step
        self.status = "completed"
        return None

    def is_complete(self):
        return self.current_step >= len(self.steps)

    def to_dict(self):
        return {
            "objective": self.objective,
            "steps": self.steps,
            "current_step": self.current_step,
            "status": self.status,
            "created_at": self.created_at
        }

    def __str__(self):
        lines = [f"🎯 Objectif: {self.objective}"]
        lines.append(f"📋 {len(self.steps)} étape(s):")
        for i, step in enumerate(self.steps):
            marker = "✅" if i < self.current_step else "⬜"
            lines.append(f"  {marker} {i+1}. {step.get('description', step.get('action', '?'))}")
        return "\n".join(lines)


class Planner:

    PLANNING_PROMPT = """Tu es Micheline, une IA agent autonome avec des OUTILS.
Tu DOIS utiliser tes outils pour accomplir les objectifs. Ne réponds JAMAIS directement sans outil.

OBJECTIF: {objective}

CONTEXTE PRÉCÉDENT:
{context}

=== OUTILS DISPONIBLES ===
{tools_description}
=== FIN OUTILS ===

RÈGLES STRICTES:
1. Tu DOIS choisir un ou plusieurs outils parmi la liste ci-dessus
2. Le champ "action" DOIT être exactement le nom d'un outil (ex: "calculator", "memory_stats", "read_file")
3. Le champ "params" DOIT contenir les paramètres requis par l'outil
4. N'utilise "respond" que si aucun outil ne correspond ET que tu as déjà un résultat
5. N'utilise "think" que pour analyser avant d'utiliser un outil
6. Maximum 5 étapes

EXEMPLES:
- "Calcule 2+2" → action: "calculator", params: {{"expression": "2+2"}}
- "Quelle heure est-il" → action: "datetime", params: {{}}
- "Stats mémoire" → action: "memory_stats", params: {{}}
- "Cherche en mémoire" → action: "memory_search", params: {{"query": "..."}}
- "Lis le fichier X" → action: "read_file", params: {{"path": "X"}}
- "Liste les dossiers" → action: "list_allowed_paths", params: {{}}
- "Infos système" → action: "system_info", params: {{}}

Réponds UNIQUEMENT en JSON valide:
{{
    "steps": [
        {{
            "action": "nom_exact_de_l_outil",
            "params": {{}},
            "description": "ce que cette étape fait"
        }}
    ]
}}"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._tools_description = ""

    def set_tools_description(self, description: str):
        """Met à jour la description des outils disponibles."""
        self._tools_description = description

    def create_plan(self, objective: str, context: str = "", available_tools: list = None) -> Plan:
        if available_tools is None:
            available_tools = ["respond", "think"]

        # D'ABORD essayer le fallback intelligent (rapide et fiable)
        fallback = self._try_fallback(objective)
        if fallback:
            print(f"[Planner] ✅ Fallback utilisé (pas besoin du LLM)")
            return Plan(objective=objective, steps=fallback.get("steps", []))

        # Si pas de fallback, utiliser le LLM
        tools_desc = self._tools_description
        if not tools_desc:
            tools_desc = "Outils: " + ", ".join(available_tools)

        prompt = self.PLANNING_PROMPT.format(
            objective=objective,
            context=context if context else "Aucun contexte.",
            tools_description=tools_desc
        )

        if self.llm_client:
            response = self._call_llm(prompt)
            plan_data = self._parse_response(response, objective, available_tools)
        else:
            plan_data = {
                "steps": [
                    {
                        "action": "respond",
                        "params": {"message": objective},
                        "description": f"Répondre à: {objective}"
                    }
                ]
            }

        return Plan(objective=objective, steps=plan_data.get("steps", []))

    def _try_fallback(self, objective: str) -> dict:
        """
        Essaie de deviner l'outil SANS appeler le LLM.
        Retourne un plan dict ou None si pas de match.
        """
        import unicodedata
        import re

        def no_accent(s):
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

        obj_norm = no_accent(objective.lower().strip())
        
                # === DEBUG ===
        print(f"[Planner DEBUG] objective brut: '{objective}'")
        print(f"[Planner DEBUG] obj_norm: '{obj_norm}'")
        print(f"[Planner DEBUG] 'en memoire' in obj_norm = {'en memoire' in obj_norm}")
        # === FIN DEBUG ===

        # Mapping mots-clés → outil
        rules = [
            # Mémoire
            (["stats memoire", "stats de ta memoire", "statistiques memoire",
              "statistiques de ta memoire"],
             "memory_stats", {}),
            (["en memoire", "dans ta memoire", "tu as en memoire",
              "as-tu en memoire", "as tu en memoire", "souviens",
              "rappelles", "remember", "tes experiences"],
             "memory_search", {"query": ""}),
            # Calcul
            (["calcule", "calculate", "compute", "racine carree", "sqrt"],
             "calculator", {"expression": self._extract_expression(objective)}),
            # Date/heure
            (["quelle heure", "quel jour", "quelle date", "date et heure",
              "date aujourd", "what time"],
             "datetime", {}),
            # Système
            (["info systeme", "infos systeme", "system info", "info system",
              "ram disponible", "quel os"],
             "system_info", {}),
            # Dossiers autorisés
            (["dossiers ou tu peux", "ou peux-tu travailler", "ou tu peux travailler",
              "dossiers autorises", "allowed paths", "liste les dossiers"],
             "list_allowed_paths", {}),
            # Lecture fichier
            (["lis le fichier", "lire le fichier", "read file", "contenu du fichier",
              "ouvre le fichier"],
             "read_file", {"path": self._extract_path(objective)}),
            # Liste fichiers
            (["liste les fichiers", "list directory", "contenu du dossier",
              "fichiers dans"],
             "list_directory", {"path": self._extract_path(objective) or "."}),
            # Info fichier
            (["info fichier", "taille du fichier", "file info"],
             "file_info", {"path": self._extract_path(objective)}),
        ]

        for keywords, tool_name, params in rules:
            for kw in keywords:
                if kw in obj_norm:
                    print(f"[Planner] Fallback match: '{kw}' → '{tool_name}'")
                    return {
                        "steps": [{
                            "action": tool_name,
                            "params": params,
                            "description": objective
                        }]
                    }

        return None

    def _call_llm(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'chat') and callable(self.llm_client.chat):
                answer, dt, usage = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="Tu es un planificateur d'actions. Réponds UNIQUEMENT en JSON valide. Utilise les outils disponibles.",
                    temperature=0.1,
                    max_tokens=1500
                )
                # DEBUG: voir ce que le LLM a produit
                print(f"[Planner] Réponse LLM brute: {(answer or '')[:300]}")
                return answer or ""
            return "{}"
        except Exception as e:
            print(f"[Planner] Erreur LLM: {e}")
            return "{}"
            
    def _parse_response(self, response: str, objective: str, available_tools: list) -> dict:
        """Parse la réponse JSON du LLM avec fallback intelligent."""
        
        # Essayer de parser le JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                if "steps" in data and isinstance(data["steps"], list) and len(data["steps"]) > 0:
                    return data
        except json.JSONDecodeError:
            pass

        # === FALLBACK INTELLIGENT ===
        # Si le LLM n'a pas produit de JSON valide, on devine l'outil
        obj_lower = objective.lower()
        
        import unicodedata
        def no_accent(s):
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        
        obj_norm = no_accent(obj_lower)

        # Mapping mots-clés → outil
        fallback_rules = [
            # Mémoire
            (["stats memoire", "stats de ta memoire", "statistiques memoire"], 
             "memory_stats", {}),
            (["en memoire", "dans ta memoire", "tu as en memoire", "souviens", "rappelles", "remember"],
             "memory_search", {"query": ""}),
            # Calcul
            (["calcule", "calculate", "compute", "racine", "sqrt"],
             "calculator", {"expression": self._extract_expression(objective)}),
            # Date/heure
            (["heure", "date", "time", "quelle heure", "quel jour"],
             "datetime", {}),
            # Système
            (["systeme", "system", "ram", "os", "python"],
             "system_info", {}),
            # Fichiers
            (["dossiers autorises", "allowed paths", "ou peux-tu", "ou tu peux"],
             "list_allowed_paths", {}),
            (["lis le fichier", "read file", "contenu du fichier"],
             "read_file", {"path": self._extract_path(objective)}),
            (["liste les fichiers", "list directory", "contenu du dossier"],
             "list_directory", {"path": "."}),
        ]

        for keywords, tool_name, params in fallback_rules:
            for kw in keywords:
                if kw in obj_norm:
                    print(f"[Planner] Fallback: '{kw}' → outil '{tool_name}'")
                    return {
                        "steps": [{
                            "action": tool_name,
                            "params": params,
                            "description": objective
                        }]
                    }

        # Dernier recours: répondre directement
        return {
            "steps": [{
                "action": "respond",
                "params": {"message": objective},
                "description": f"Répondre à: {objective}"
            }]
        }

    def _extract_expression(self, text: str) -> str:
        """Extrait une expression mathématique du texte."""
        import re
        # Cherche des patterns numériques
        patterns = [
            r"(\d+[\s]*[+\-*/^][\s]*\d+)",  # 2+2, 10 * 5
            r"(sqrt\([^)]+\))",               # sqrt(144)
            r"(racine carr[ée]e? de \d+)",     # racine carrée de 144
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                expr = m.group(1)
                # Convertir "racine carrée de X" en "sqrt(X)"
                rm = re.search(r"racine carr[ée]e? de (\d+)", expr, re.IGNORECASE)
                if rm:
                    return f"sqrt({rm.group(1)})"
                return expr
        
        # Fallback: retourner tout ce qui ressemble à un calcul
        import re
        nums = re.findall(r"[\d+\-*/^().]+", text)
        return " ".join(nums) if nums else text

    def _extract_path(self, text: str) -> str:
        """Extrait un chemin de fichier du texte."""
        import re
        # Windows paths
        m = re.search(r'([A-Za-z]:\\[^\s"\']+)', text)
        if m:
            return m.group(1)
        # Unix paths
        m = re.search(r'(/[^\s"\']+)', text)
        if m:
            return m.group(1)
        # Quoted paths
        m = re.search(r'["\']([^"\']+)["\']', text)
        if m:
            return m.group(1)
        return ""