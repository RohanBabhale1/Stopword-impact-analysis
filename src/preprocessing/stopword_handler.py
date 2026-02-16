from typing import List, Set, Optional
import nltk
from nltk.corpus import stopwords

class StopwordHandler:
    """Manage stopword lists and removal strategies"""
    
    def __init__(self, language='english'):
        self.language = language
        nltk.download('stopwords', quiet=True)
        self.nltk_stopwords = set(stopwords.words(language))
        
    def get_stopwords(self, source='nltk') -> Set[str]:
        """Get stopword list from various sources"""
        if source == 'nltk':
            return self.nltk_stopwords
        elif source == 'custom':
            return self._create_custom_stopwords()
        elif source == 'minimal':
            return self._minimal_stopwords()
        elif source == 'extended':
            return self._extended_stopwords()
        else:
            return set()
    
    def _minimal_stopwords(self) -> Set[str]:
        """Minimal stopword list (articles, pronouns)"""
        return {'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    def _extended_stopwords(self) -> Set[str]:
        """Extended stopword list including NLTK + custom"""
        extended = self.nltk_stopwords.copy()
        # Add domain-specific stopwords for Reuters
        extended.update(['said', 'would', 'could', 'also', 'one', 'two'])
        return extended
    
    def _create_custom_stopwords(self) -> Set[str]:
        """Create custom stopword list based on frequency analysis"""
        # This would be populated based on corpus analysis
        return self.nltk_stopwords
    
    def remove_stopwords(self, tokens: List[str], 
                        stopword_source='nltk') -> List[str]:
        """Remove stopwords from token list"""
        stop_words = self.get_stopwords(stopword_source)
        return [token for token in tokens if token not in stop_words]
    
    def analyze_stopword_presence(self, tokens: List[str]) -> dict:
        """Analyze stopword statistics in token list"""
        stop_words = self.nltk_stopwords
        total_tokens = len(tokens)
        stopword_count = sum(1 for token in tokens if token in stop_words)
        
        return {
            'total_tokens': total_tokens,
            'stopword_count': stopword_count,
            'stopword_ratio': stopword_count / total_tokens if total_tokens > 0 else 0,
            'unique_stopwords': len(set(tokens) & stop_words)
        }