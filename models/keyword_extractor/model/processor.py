import spacy
from typing import List, Tuple, Dict, Set
from collections import Counter
import numpy as np
import re

class KeywordProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load('ru_core_news_sm')
        except OSError:
            import os
            os.system('python -m spacy download ru_core_news_sm')
            self.nlp = spacy.load('ru_core_news_sm')
            
        # Веса для разных частей речи
        self.pos_weights = {
            'NOUN': 1.0,
            'ADJ': 0.8,
            'VERB': 0.6
        }
        
        # Тематически важные слова
        self.domain_words = {
            'интеллект': 1.5,
            'система': 1.3,
            'алгоритм': 1.3,
            'обучение': 1.3,
            'анализ': 1.2,
            'технология': 1.2,
            'данные': 1.2,
            'модель': 1.2,
            'распознавание': 1.2,
            'компьютер': 1.2
        }
    
    def _calculate_word_importance(self, token) -> float:
        base_weight = 1.0
        pos_weight = self.pos_weights.get(token.pos_, 0.5)
        domain_weight = self.domain_words.get(token.lemma_, 1.0)
        entity_weight = 1.2 if token.ent_type_ else 1.0
        dep_weight = 1.2 if token.dep_ in {'ROOT', 'nsubj', 'dobj'} else 1.0
        return base_weight * pos_weight * domain_weight * entity_weight * dep_weight

    def _agree_words(self, token1, token2) -> str:
        return f"{token1.text} {token2.text}"
            
    def _get_word_frequencies(self, doc) -> Dict[str, float]:
        word_scores = {}
        sentence_count = 0
        
        for sent in doc.sents:
            sentence_count += 1
            for i, token in enumerate(sent):
                if (token.pos_ in self.pos_weights and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) >= 3):
                    
                    importance = self._calculate_word_importance(token)
                    sent_len = len(list(sent))
                    position_weight = 1.2 if i < 3 else 1.1 if i >= sent_len - 3 else 1.0
                    sent_weight = 1.2 if sentence_count == 1 else 1.0
                    score = importance * position_weight * sent_weight
                    
                    if token.text in word_scores:
                        word_scores[token.text] += score * 0.8
                    else:
                        word_scores[token.text] = score
                        
        return word_scores
        
    def _is_valid_bigram(self, token1, token2) -> Tuple[bool, float]:
        weight = 1.0
        if token2.dep_ in {'amod', 'compound', 'nmod'}:
            weight *= 1.2
        
        valid_combinations = {
            ('ADJ', 'NOUN'): 1.3,
            ('NOUN', 'NOUN'): 1.2,
            ('NOUN', 'ADJ'): 1.1
        }
        
        pos_combo = (token1.pos_, token2.pos_)
        if pos_combo in valid_combinations:
            weight *= valid_combinations[pos_combo]
            return True, weight
            
        return False, 0.0
        
    def _get_bigrams(self, doc) -> Dict[str, float]:
        bigram_scores = {}
        seen_bigrams = set()
        
        for sent in doc.sents:
            sent_tokens = list(sent)
            for i in range(len(sent_tokens) - 1):
                token1 = sent_tokens[i]
                token2 = sent_tokens[i + 1]
                
                if (not token1.is_stop and not token2.is_stop and
                    not token1.is_punct and not token2.is_punct and
                    len(token1.text) >= 3 and len(token2.text) >= 3):
                    
                    is_valid, syntax_weight = self._is_valid_bigram(token1, token2)
                    
                    if is_valid:
                        bigram = self._agree_words(token1, token2)
                        
                        if bigram not in seen_bigrams:
                            word1_importance = self._calculate_word_importance(token1)
                            word2_importance = self._calculate_word_importance(token2)
                            importance = (word1_importance + word2_importance) / 2
                            score = importance * syntax_weight
                            
                            bigram_scores[bigram] = score
                            seen_bigrams.add(bigram)
                            
        return bigram_scores

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        doc = self.nlp(text.lower())
        
        word_scores = self._get_word_frequencies(doc)
        bigram_scores = self._get_bigrams(doc)
        
        print("\nОтладка - веса слов:")
        print(dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]))
        print("\nОтладка - веса биграмм:")
        print(dict(sorted(bigram_scores.items(), key=lambda x: x[1], reverse=True)[:5]))
        
        all_keywords = {}
        
        if word_scores:
            max_word_score = max(word_scores.values())
            for word, score in word_scores.items():
                all_keywords[word] = score / max_word_score
        
        if bigram_scores:
            max_bigram_score = max(bigram_scores.values())
            for bigram, score in bigram_scores.items():
                all_keywords[bigram] = (score / max_bigram_score) * 1.2
        
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:num_keywords]
