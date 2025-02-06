# model/utils/analysis.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import logging
from pathlib import Path
import json
from datetime import datetime

from .metrics import KeywordMetrics
from ..config.logging_config import get_logger

logger = get_logger(__name__)

class ErrorAnalyzer:
    """Анализ ошибок модели"""
    
    def __init__(self):
        self.error_stats = defaultdict(list)
        
    def analyze_predictions(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        targets: List[str],
        texts: List[str]
    ) -> Dict[str, Dict]:
        """
        Анализ предсказаний модели
        
        Args:
            predictions: Предсказанные ключевые слова с оценками
            targets: Целевые ключевые слова
            texts: Исходные тексты
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            analysis = {
                'error_types': self._analyze_error_types(predictions, targets),
                'length_analysis': self._analyze_length_impact(predictions, targets, texts),
                'confidence_analysis': self._analyze_confidence(predictions),
                'context_analysis': self._analyze_context(predictions, texts)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка при анализе предсказаний: {e}")
            raise
            
    def _analyze_error_types(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        targets: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Анализ типов ошибок"""
        error_types = defaultdict(int)
        total_samples = len(predictions)
        
        for pred, target in zip(predictions, targets):
            pred_keywords = {p['keyword'] for p in pred}
            target_keywords = set(target)
            
            # Ложноположительные
            false_positives = pred_keywords - target_keywords
            if false_positives:
                error_types['false_positives'] += len(false_positives)
                
            # Ложноотрицательные
            false_negatives = target_keywords - pred_keywords
            if false_negatives:
                error_types['false_negatives'] += len(false_negatives)
                
            # Частично правильные
            partial_matches = sum(
                any(t in p['keyword'] for p in pred)
                for t in target_keywords
            )
            if partial_matches:
                error_types['partial_matches'] += partial_matches
                
        # Нормализация статистики
        stats = {
            error_type: {
                'count': count,
                'percentage': (count / total_samples) * 100
            }
            for error_type, count in error_types.items()
        }
        
        return stats
        
    def _analyze_length_impact(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        targets: List[str],
        texts: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Анализ влияния длины текста"""
        length_stats = defaultdict(list)
        
        for pred, target, text in zip(predictions, targets, texts):
            text_length = len(text.split())
            pred_keywords = {p['keyword'] for p in pred}
            target_keywords = set(target)
            
            # Точность для текста
            precision = len(pred_keywords & target_keywords) / len(pred_keywords) if pred_keywords else 0
            
            # Группировка по длине
            length_bucket = text_length // 100 * 100  # Группы по 100 слов
            length_stats[length_bucket].append(precision)
            
        # Агрегация статистики
        return {
            length: {
                'avg_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'count': len(precisions)
            }
            for length, precisions in length_stats.items()
        }
        
    def _analyze_confidence(
        self,
        predictions: List[Dict[str, Union[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Анализ уверенности предсказаний"""
        confidence_stats = defaultdict(list)
        
        for pred in predictions:
            for p in pred:
                confidence_bucket = round(p['score'] * 10) / 10  # Группы по 0.1
                confidence_stats[confidence_bucket].append(p['keyword'])
                
        return {
            conf: {
                'count': len(keywords),
                'unique_count': len(set(keywords)),
                'avg_length': np.mean([len(k) for k in keywords])
            }
            for conf, keywords in confidence_stats.items()
        }
        
    def _analyze_context(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        texts: List[str]
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Анализ контекста ключевых слов"""
        context_analysis = []
        
        for pred, text in zip(predictions, texts):
            words = text.split()
            for p in pred:
                keyword = p['keyword']
                # Поиск позиции ключевого слова
                try:
                    idx = words.index(keyword)
                    # Получение контекстного окна
                    context = ' '.join(
                        words[max(0, idx-3):min(len(words), idx+4)]
                    )
                    context_analysis.append({
                        'keyword': keyword,
                        'score': p['score'],
                        'context': context,
                        'position': idx / len(words)
                    })
                except ValueError:
                    continue
                    
        return context_analysis

class PerformanceAnalyzer:
    """Анализ производительности модели"""
    
    def __init__(self):
        self.metrics = KeywordMetrics()
        
    def analyze_model_performance(
        self,
        model_outputs: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Анализ производительности модели
        
        Args:
            model_outputs: Выходы модели
            targets: Целевые значения
            
        Returns:
            Словарь с метриками производительности
        """
        try:
            performance = {
                'overall_metrics': self._calculate_overall_metrics(
                    model_outputs,
                    targets
                ),
                'per_length_metrics': self._analyze_by_length(
                    model_outputs,
                    targets
                ),
                'threshold_analysis': self._analyze_threshold_impact(
                    model_outputs,
                    targets
                )
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Ошибка при анализе производительности: {e}")
            raise
            
    def _calculate_overall_metrics(
        self,
        outputs: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Расчет общих метрик"""
        all_predictions = np.concatenate(
            [o['predictions'] for o in outputs]
        )
        all_targets = np.concatenate(
            [t['labels'] for t in targets]
        )
        
        return self.metrics.calculate_metrics(
            torch.from_numpy(all_predictions),
            torch.from_numpy(all_targets)
        )
        
    def _analyze_by_length(
        self,
        outputs: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, float]]:
        """Анализ метрик в зависимости от длины"""
        length_metrics = defaultdict(list)
        
        for output, target in zip(outputs, targets):
            length = len(output['predictions'])
            metrics = self.metrics.calculate_metrics(
                torch.from_numpy(output['predictions']),
                torch.from_numpy(target['labels'])
            )
            length_metrics[length].append(metrics)
            
        return {
            length: {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0].keys()
            }
            for length, metrics_list in length_metrics.items()
        }
        
    def _analyze_threshold_impact(
        self,
        outputs: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, float]]:
        """Анализ влияния порога уверенности"""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
            
        threshold_metrics = {}
        all_predictions = np.concatenate(
            [o['predictions'] for o in outputs]
        )
        all_targets = np.concatenate(
            [t['labels'] for t in targets]
        )
        
        for threshold in thresholds:
            binary_predictions = (all_predictions >= threshold).astype(int)
            metrics = self.metrics.calculate_metrics(
                torch.from_numpy(binary_predictions),
                torch.from_numpy(all_targets)
            )
            threshold_metrics[float(threshold)] = metrics
            
        return threshold_metrics
