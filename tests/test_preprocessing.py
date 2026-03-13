"""
Unit Tests: src/preprocessing/
Tests TextCleaner, StopwordHandler, and AdaptiveStopwordHandler
Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.stopword_handler import StopwordHandler
from src.preprocessing.adaptive_stopwords import AdaptiveStopwordHandler


# ============================================================
# TextCleaner Tests
# ============================================================

class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner()

    # --- clean() ---

    def test_clean_lowercases_text(self):
        result = self.cleaner.clean("Reuters SAID Earnings ROSE")
        assert result == result.lower()

    def test_clean_removes_http_url(self):
        result = self.cleaner.clean("Visit http://reuters.com for details")
        assert "http" not in result
        assert "reuters.com" not in result

    def test_clean_removes_www_url(self):
        result = self.cleaner.clean("See www.example.com for more")
        assert "www" not in result

    def test_clean_removes_email(self):
        result = self.cleaner.clean("Contact editor@reuters.com today")
        assert "@" not in result

    def test_clean_removes_punctuation(self):
        result = self.cleaner.clean("Hello, world! Prices rose.")
        for char in ".,!?;:":
            assert char not in result

    def test_clean_removes_numbers_by_default(self):
        result = self.cleaner.clean("Earnings rose 25 pct in 2024")
        assert not any(c.isdigit() for c in result)

    def test_clean_preserves_numbers_when_disabled(self):
        cleaner = TextCleaner(remove_numbers=False)
        result = cleaner.clean("Revenue was 500 mln dlrs")
        assert "500" in result

    def test_clean_empty_string_returns_empty(self):
        assert self.cleaner.clean("") == ""

    def test_clean_none_returns_empty(self):
        assert self.cleaner.clean(None) == ""

    def test_clean_collapses_whitespace(self):
        result = self.cleaner.clean("hello   world   test")
        assert "  " not in result

    def test_clean_preserves_content_words(self):
        result = self.cleaner.clean("crude oil prices rose sharply")
        for word in ["crude", "oil", "prices", "rose", "sharply"]:
            assert word in result

    def test_clean_no_lowercase_option(self):
        cleaner = TextCleaner(lowercase=False)
        result = cleaner.clean("Reuters")
        assert "Reuters" in result

    def test_clean_returns_string(self):
        result = self.cleaner.clean("some text here")
        assert isinstance(result, str)

    # --- tokenize_and_process() ---

    def test_tokenize_returns_list(self):
        result = self.cleaner.tokenize_and_process("crude oil market")
        assert isinstance(result, list)

    def test_tokenize_splits_into_words(self):
        result = self.cleaner.tokenize_and_process("crude oil market")
        assert "crude" in result
        assert "oil" in result
        assert "market" in result

    def test_tokenize_empty_string_returns_list(self):
        result = self.cleaner.tokenize_and_process("")
        assert isinstance(result, list)

    def test_tokenize_with_stemming_reduces_forms(self):
        cleaner = TextCleaner(stemming=True)
        result = cleaner.tokenize_and_process("earnings earning earned")
        assert len(set(result)) < 3

    def test_tokenize_with_lemmatization_returns_list(self):
        cleaner = TextCleaner(lemmatization=True)
        result = cleaner.tokenize_and_process("running runs ran")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_full_pipeline_clean_then_tokenize(self):
        text = "The Company SAID it earned 50 mln dlrs in Q3!"
        cleaned = self.cleaner.clean(text)
        tokens = self.cleaner.tokenize_and_process(cleaned)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "50" not in tokens
        assert "!" not in tokens


# ============================================================
# StopwordHandler Tests
# ============================================================

class TestStopwordHandler:

    def setup_method(self):
        self.handler = StopwordHandler()

    # --- get_stopwords() ---

    def test_get_nltk_returns_set(self):
        assert isinstance(self.handler.get_stopwords('nltk'), set)

    def test_get_nltk_not_empty(self):
        assert len(self.handler.get_stopwords('nltk')) > 0

    def test_get_nltk_contains_common_words(self):
        sw = self.handler.get_stopwords('nltk')
        for word in ['the', 'is', 'at', 'on', 'a']:
            assert word in sw

    def test_get_minimal_returns_set(self):
        assert isinstance(self.handler.get_stopwords('minimal'), set)

    def test_get_minimal_is_small(self):
        assert len(self.handler.get_stopwords('minimal')) <= 15

    def test_get_minimal_contains_articles(self):
        assert 'the' in self.handler.get_stopwords('minimal')

    def test_get_extended_superset_of_nltk(self):
        nltk_sw = self.handler.get_stopwords('nltk')
        extended_sw = self.handler.get_stopwords('extended')
        assert nltk_sw.issubset(extended_sw)

    def test_get_extended_contains_domain_words(self):
        sw = self.handler.get_stopwords('extended')
        for word in ['said', 'would', 'could']:
            assert word in sw

    def test_get_none_returns_empty_set(self):
        assert self.handler.get_stopwords('none') == set()

    def test_extended_larger_than_nltk(self):
        assert len(self.handler.get_stopwords('extended')) > len(self.handler.get_stopwords('nltk'))

    def test_minimal_smaller_than_nltk(self):
        assert len(self.handler.get_stopwords('minimal')) < len(self.handler.get_stopwords('nltk'))

    # --- remove_stopwords() ---

    def test_remove_stopwords_returns_list(self):
        tokens = ['the', 'crude', 'oil', 'market', 'is', 'rising']
        result = self.handler.remove_stopwords(tokens, stopword_source='nltk')
        assert isinstance(result, list)

    def test_remove_stopwords_nltk_removes_the(self):
        tokens = ['the', 'crude', 'oil']
        result = self.handler.remove_stopwords(tokens, stopword_source='nltk')
        assert 'the' not in result

    def test_remove_stopwords_preserves_content_words(self):
        tokens = ['the', 'crude', 'oil', 'market']
        result = self.handler.remove_stopwords(tokens, stopword_source='nltk')
        assert 'crude' in result
        assert 'oil' in result
        assert 'market' in result

    def test_remove_stopwords_result_subset_of_input(self):
        tokens = ['the', 'crude', 'oil', 'market', 'is']
        result = self.handler.remove_stopwords(tokens, stopword_source='nltk')
        assert all(t in tokens for t in result)

    def test_remove_stopwords_empty_list(self):
        result = self.handler.remove_stopwords([], stopword_source='nltk')
        assert result == []

    def test_remove_stopwords_extended_removes_said(self):
        tokens = ['company', 'said', 'earnings', 'rose']
        result = self.handler.remove_stopwords(tokens, stopword_source='extended')
        assert 'said' not in result
        assert 'company' in result

    # --- analyze_stopword_presence() ---

    def test_analyze_returns_dict(self):
        tokens = ['the', 'crude', 'oil', 'is', 'rising']
        result = self.handler.analyze_stopword_presence(tokens)
        assert isinstance(result, dict)

    def test_analyze_has_required_keys(self):
        tokens = ['the', 'crude', 'oil']
        result = self.handler.analyze_stopword_presence(tokens)
        for key in ['total_tokens', 'stopword_count', 'stopword_ratio', 'unique_stopwords']:
            assert key in result

    def test_analyze_total_tokens_correct(self):
        tokens = ['the', 'crude', 'oil']
        result = self.handler.analyze_stopword_presence(tokens)
        assert result['total_tokens'] == 3

    def test_analyze_ratio_between_0_and_1(self):
        tokens = ['the', 'crude', 'oil', 'is', 'rising']
        result = self.handler.analyze_stopword_presence(tokens)
        assert 0.0 <= result['stopword_ratio'] <= 1.0

    def test_analyze_empty_tokens_no_crash(self):
        result = self.handler.analyze_stopword_presence([])
        assert result['stopword_ratio'] == 0


# ============================================================
# AdaptiveStopwordHandler Tests
# ============================================================

class TestAdaptiveStopwordHandler:

    def setup_method(self):
        self.handler = AdaptiveStopwordHandler(threshold=0.0005)  
        self.corpus = [
            "crude oil prices rose sharply today",
            "oil market expects further gains next week",
            "the company said earnings rose last quarter",
            "trade deficit widened as imports rose sharply",
            "grain prices fell amid bumper harvest season",
            "stock market rallied on strong earnings report",
            "dollar weakened against major currencies today",
            "interest rates expected to rise next month",
        ] * 10

    def test_generate_returns_list_or_set(self):
        result = self.handler.generate(self.corpus)
        assert isinstance(result, (list, set))

    def test_generate_not_empty(self):
        """Use high threshold to guarantee words are selected on small corpus"""
        handler = AdaptiveStopwordHandler(threshold=0.5)
        result = handler.generate(self.corpus)
        assert len(result) > 0

    def test_generate_higher_threshold_gives_larger_set(self):
        low_handler  = AdaptiveStopwordHandler(threshold=0.0001)
        high_handler = AdaptiveStopwordHandler(threshold=0.001)
        low_result   = low_handler.generate(self.corpus)
        high_result  = high_handler.generate(self.corpus)
        assert len(high_result) >= len(low_result)

    def test_generate_empty_corpus_no_crash(self):
        try:
            result = self.handler.generate([])
            assert isinstance(result, (list, set))
        except ValueError:
            pass 

    def test_generate_single_doc_no_crash(self):
        result = self.handler.generate(["crude oil prices rose"])
        assert isinstance(result, (list, set))

    def test_threshold_stored_correctly(self):
        handler = AdaptiveStopwordHandler(threshold=0.005)
        assert handler.threshold == 0.005

    def test_generate_result_contains_strings(self):
        result = self.handler.generate(self.corpus)
        for item in result:
            assert isinstance(item, str)