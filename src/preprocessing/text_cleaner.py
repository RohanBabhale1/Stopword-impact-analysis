import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class TextCleaner:
    """Clean and preprocess text data"""
    
    def __init__(self, 
                 lowercase=True,
                 remove_punctuation=True,
                 remove_numbers=True,
                 stemming=False,
                 lemmatization=False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        if stemming:
            self.stemmer = PorterStemmer()
        if lemmatization:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean(self, text: str) -> str:
        """Apply cleaning pipeline to text"""
        if not text:
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """Tokenize and optionally stem/lemmatize"""
        tokens = word_tokenize(text)
        
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens