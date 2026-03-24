"""
Web Search Tool — Recherche d'informations sur le web.
Emplacement : micheline/tools/web_search_tool.py
"""

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, List
from datetime import datetime


class WebSearchEngine:
    """Moteur de recherche web multi-sources."""
    
    def __init__(self):
        self.news_api_key = None
        self._load_config()
    
    def _load_config(self):
        """Charge les clés API depuis la config."""
        try:
            import yaml
            import os
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'news' in config:
                        self.news_api_key = config['news'].get('api_key')
        except Exception:
            pass
        
        # Fallback : essayer de lire depuis les variables d'environnement
        if not self.news_api_key:
            import os
            self.news_api_key = os.environ.get('NEWS_API_KEY')
    
    def search_news(self, query: str, language: str = "fr", max_results: int = 5) -> List[Dict]:
        """Recherche via NewsAPI."""
        if not self.news_api_key:
            return [{"error": "Clé NewsAPI non configurée. Ajoute 'NEWS_API_KEY' dans settings.yaml ou les variables d'environnement."}]
        
        try:
            params = urllib.parse.urlencode({
                'q': query,
                'language': language,
                'pageSize': max_results,
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            })
            url = f"https://newsapi.org/v2/everything?{params}"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'MichelineBot/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('status') != 'ok':
                return [{"error": f"NewsAPI erreur : {data.get('message', 'inconnue')}"}]
            
            results = []
            for article in data.get('articles', [])[:max_results]:
                results.append({
                    "title": article.get('title', 'Sans titre'),
                    "description": article.get('description', 'Pas de description'),
                    "source": article.get('source', {}).get('name', 'Inconnu'),
                    "url": article.get('url', ''),
                    "published": article.get('publishedAt', ''),
                    "content_preview": (article.get('content', '') or '')[:200]
                })
            
            return results
            
        except urllib.error.URLError as e:
            return [{"error": f"Erreur réseau : {e}"}]
        except Exception as e:
            return [{"error": f"Erreur inattendue : {e}"}]
    
    def search_wikipedia(self, query: str, language: str = "fr", max_results: int = 3) -> List[Dict]:
        """Recherche sur Wikipedia (API gratuite, pas de clé nécessaire)."""
        try:
            # Étape 1 : Rechercher les titres
            params = urllib.parse.urlencode({
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'format': 'json',
                'utf8': 1
            })
            url = f"https://{language}.wikipedia.org/w/api.php?{params}"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'MichelineBot/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            results = []
            for item in data.get('query', {}).get('search', []):
                # Nettoyer le snippet HTML
                snippet = item.get('snippet', '')
                # Retirer les tags HTML basiques
                import re
                snippet = re.sub(r'<[^>]+>', '', snippet)
                snippet = snippet.replace('&#039;', "'").replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>')
                
                results.append({
                    "title": item.get('title', ''),
                    "description": snippet,
                    "source": f"Wikipedia ({language})",
                    "url": f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
                    "word_count": item.get('wordcount', 0)
                })
            
            return results
            
        except Exception as e:
            return [{"error": f"Erreur Wikipedia : {e}"}]
    
    def get_wikipedia_summary(self, title: str, language: str = "fr") -> str:
        """Obtient le résumé complet d'un article Wikipedia."""
        try:
            params = urllib.parse.urlencode({
                'action': 'query',
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'titles': title,
                'format': 'json',
                'utf8': 1
            })
            url = f"https://{language}.wikipedia.org/w/api.php?{params}"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'MichelineBot/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id != '-1':
                    return page_data.get('extract', 'Aucun contenu trouvé.')
            
            return "Article non trouvé."
            
        except Exception as e:
            return f"Erreur : {e}"


# Instance globale
_engine = WebSearchEngine()


def web_search(query: str, source: str = "all", language: str = "fr", max_results: int = 5) -> str:
    """
    Point d'entrée pour le tool registry.
    
    Args:
        query: Termes de recherche
        source: "news", "wikipedia", "wiki_summary", ou "all"
        language: Code langue (fr, en, etc.)
        max_results: Nombre max de résultats
    
    Returns:
        Résultats formatés en texte
    """
    if not query or not query.strip():
        return "Erreur : aucun terme de recherche fourni."
    
    query = query.strip()
    parts = []
    
    def clean_html(text: str) -> str:
        """Nettoie les entités HTML d'un texte."""
        if not text:
            return text
        text = text.replace('&#039;', "'")
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&#39;', "'")
        text = text.replace('&apos;', "'")
        import re
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    if source in ("all", "news"):
        news_results = _engine.search_news(query, language, max_results)
        if news_results and "error" not in news_results[0]:
            parts.append("📰 ACTUALITÉS :")
            for i, article in enumerate(news_results, 1):
                title = clean_html(article.get('title', 'Sans titre'))
                parts.append(f"\n  {i}. {title}")
                parts.append(f"     Source : {article.get('source', 'Inconnu')}")
                parts.append(f"     Date : {article['published'][:10] if article.get('published') else 'N/A'}")
                if article.get('description'):
                    desc = clean_html(article['description'])
                    parts.append(f"     {desc[:150]}")
                if article.get('url'):
                    parts.append(f"     🔗 {article['url']}")
        elif news_results and "error" in news_results[0]:
            parts.append(f"📰 News : {news_results[0]['error']}")
    
    if source in ("all", "wikipedia"):
        wiki_results = _engine.search_wikipedia(query, language, max_results)
        if wiki_results and "error" not in wiki_results[0]:
            parts.append("\n📚 WIKIPEDIA :")
            for i, article in enumerate(wiki_results, 1):
                title = clean_html(article.get('title', ''))
                parts.append(f"\n  {i}. {title}")
                if article.get('description'):
                    desc = clean_html(article['description'])
                    parts.append(f"     {desc[:200]}")
                if article.get('url'):
                    parts.append(f"     🔗 {article['url']}")
        elif wiki_results and "error" in wiki_results[0]:
            parts.append(f"\n📚 Wikipedia : {wiki_results[0]['error']}")
    
    if source == "wiki_summary":
        summary = _engine.get_wikipedia_summary(query, language)
        summary = clean_html(summary)
        parts.append(f"📚 Résumé Wikipedia — {query} :\n")
        if len(summary) > 2000:
            parts.append(summary[:2000] + "\n... [tronqué]")
        else:
            parts.append(summary)
    
    if not parts:
        return f"Aucun résultat trouvé pour '{query}'."
    
    return "\n".join(parts)

# Métadonnées pour le registry
TOOL_NAME = "web_search"
TOOL_DESCRIPTION = (
    "Recherche des informations sur le web. "
    "Peut chercher dans les actualités (NewsAPI) et Wikipedia. "
    "Sources disponibles : 'news' (actualités), 'wikipedia' (articles), "
    "'wiki_summary' (résumé d'un article précis), 'all' (tout)."
)
TOOL_PARAMETERS = {
    "query": "str — Termes de recherche",
    "source": "str — 'news', 'wikipedia', 'wiki_summary', ou 'all' (défaut: 'all')",
    "language": "str — Code langue : 'fr', 'en', etc. (défaut: 'fr')",
    "max_results": "int — Nombre de résultats (défaut: 5)"
}