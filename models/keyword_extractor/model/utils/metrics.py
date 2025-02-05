# model/utils/metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import logging
from collections import defaultdict

from ..config.logging_config import get_logger

logger = get_logger(__name__)

class KeywordMetrics:
    """Класс для расчета метрик качества извлечения ключевых слов"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Расчет метрик для предсказаний
        
        Args:
            predictions: Предсказания модели (после softmax)
            targets: Целевые значения
            mask: Маска для валидных токенов
            
        Returns:
            Словарь с метриками
        """
        try:
            # Применяем маску если есть
            if mask is not None:
                predictions = predictions[mask]
                targets = targets[mask]
            
            # Преобразование в numpy
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Бинаризация предсказаний
            pred_labels = (predictions >= 0.5).astype(int)
            
            # Расчет базовых метрик
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets,
                pred_labels,
                average='binary'
            )
            
            # Расчет AP и accuracy
            ap = average_precision_score(targets, predictions)
            accuracy = (pred_labels == targets).mean()
            
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'average_precision': float(ap)
            }
            
            # Сохранение в историю
            for name, value in metrics.items():
                self.metrics_history[name].append(value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка при расчете метрик: {e}")
            raise
            
    def calculate_trend_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Расчет метрик для предсказания трендов
        
        Args:
            predictions: Предсказанные оценки трендов
            targets: Целевые значения трендов
            
        Returns:
            Словарь с метриками
        """
        try:
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Среднеквадратичная ошибка
            mse = np.mean((predictions - targets) ** 2)
            
            # Средняя абсолютная ошибка
            mae = np.mean(np.abs(predictions - targets))
            
            # Коэффициент корреляции
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            
            metrics = {
                'trend_mse': float(mse),
                'trend_mae': float(mae),
                'trend_correlation': float(correlation)
            }
            
            for name, value in metrics.items():
                self.metrics_history[name].append(value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка при расчете метрик трендов: {e}")
            raise
            
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Получение усредненных метрик
        
        Returns:
            Словарь с усредненными метриками
        """
        return {
            name: float(np.mean(values))
            for name, values in self.metrics_history.items()
        }
        
    def reset_metrics(self):
        """Сброс истории метрик"""
        self.metrics_history.clear()
        
class KeywordEvaluator:
    """Класс для комплексной оценки модели"""
    
    def __init__(self):
        self.metrics = KeywordMetrics()
        
    def evaluate_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Оценка результатов для одного батча
        
        Args:
            outputs: Выходы модели
            targets: Целевые значения
            
        Returns:
            Словарь с метриками
        """
        try:
            # Оценка предсказания ключевых слов
            keyword_metrics = self.metrics.calculate_metrics(
                torch.softmax(outputs['keyword_logits'], dim=-1)[:, :, 1],
                targets['keyword_labels'],
                targets['attention_mask']
            )
            
            # Оценка предсказания трендов
            trend_metrics = self.metrics.calculate_trend_metrics(
                outputs['trend_scores'],
                targets['trend_labels']
            )
            
            return {**keyword_metrics, **trend_metrics}
            
        except Exception as e:
            logger.error(f"Ошибка при оценке батча: {e}")
            raise
            
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> Dict[str, float]:
        """
        Полная оценка модели
        
        Args:
            model: Модель для оценки
            dataloader: Загрузчик данных
            device: Устройство для вычислений
            
        Returns:
            Словарь с метриками
        """
        model.eval()
        self.metrics.reset_metrics()
      # model/utils/metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import logging
from collections import defaultdict

from ..config.logging_config import get_logger

logger = get_logger(__name__)

class KeywordMetrics:
    """Класс для расчета метрик качества извлечения ключевых слов"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Расчет метрик для предсказаний
        
        Args:
            predictions: Предсказания модели (после softmax)
            targets: Целевые значения
            mask: Маска для валидных токенов
            
        Returns:
            Словарь с метриками
        """
        try:
            # Применяем маску если есть
            if mask is not None:
                predictions = predictions[mask]
                targets = targets[mask]
            
            # Преобразование в numpy
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Бинаризация предсказаний
            pred_labels = (predictions >= 0.5).astype(int)
            
            # Расчет базовых метрик
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets,
                pred_labels,
                average='binary'
            )
            
            # Расчет AP и accuracy
            ap = average_precision_score(targets, predictions)
            accuracy = (pred_labels == targets).mean()
            
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'average_precision': float(ap)
            }
            
            # Сохранение в историю
            for name, value in metrics.items():
                self.metrics_history[name].append(value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка при расчете метрик: {e}")
            raise
            
    def calculate_trend_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Расчет метрик для предсказания трендов
        
        Args:
            predictions: Предсказанные оценки трендов
            targets: Целевые значения трендов
            
        Returns:
            Словарь с метриками
        """
        try:
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Среднеквадратичная ошибка
            mse = np.mean((predictions - targets) ** 2)
            
            # Средняя абсолютная ошибка
            mae = np.mean(np.abs(predictions - targets))
            
            # Коэффициент корреляции
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            
            metrics = {
                'trend_mse': float(mse),
                'trend_mae': float(mae),
                'trend_correlation': float(correlation)
            }
            
            for name, value in metrics.items():
                self.metrics_history[name].append(value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка при расчете метрик трендов: {e}")
            raise
            
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Получение усредненных метрик
        
        Returns:
            Словарь с усредненными метриками
        """
        return {
            name: float(np.mean(values))
            for name, values in self.metrics_history.items()
        }
        
    def reset_metrics(self):
        """Сброс истории метрик"""
        self.metrics_history.clear()
        
class KeywordEvaluator:
    """Класс для комплексной оценки модели"""
    
    def __init__(self):
        self.metrics = KeywordMetrics()
        
    def evaluate_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Оценка результатов для одного батча
        
        Args:
            outputs: Выходы модели
            targets: Целевые значения
            
        Returns:
            Словарь с метриками
        """
        try:
            # Оценка предсказания ключевых слов
            keyword_metrics = self.metrics.calculate_metrics(
                torch.softmax(outputs['keyword_logits'], dim=-1)[:, :, 1],
                targets['keyword_labels'],
                targets['attention_mask']
            )
            
            # Оценка предсказания трендов
            trend_metrics = self.metrics.calculate_trend_metrics(
                outputs['trend_scores'],
                targets['trend_labels']
            )
            
            return {**keyword_metrics, **trend_metrics}
            
        except Exception as e:
            logger.error(f"Ошибка при оценке батча: {e}")
            raise
            
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> Dict[str, float]:
        """
        Полная оценка модели
        
        Args:
            model: Модель для оценки
            dataloader: Загрузчик данных
            device: Устройство для вычислений
            
        Returns:
            Словарь с метриками
        """
        model.eval()
        self.metrics.reset_metrics()
        
        total_metrics = defaultdict(float)
        num_batches = len(dataloader)
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    # Перемещение данных на устройство
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Получение предсказаний модели
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    # Расчет метрик для батча
                    batch_metrics = self.evaluate_batch(outputs, batch)
                    
                    # Накопление метрик
                    for name, value in batch_metrics.items():
                        total_metrics[name] += value
                        
            # Усреднение метрик
            avg_metrics = {
                name: value / num_batches
                for name, value in total_metrics.items()
            }
            
            # Добавление агрегированных метрик
            avg_metrics.update(self.metrics.get_average_metrics())
            
            logger.info("Оценка модели завершена")
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            raise
            
    def evaluate_keywords(
        self,
        predicted_keywords: List[str],
        target_keywords: List[str]
    ) -> Dict[str, float]:
        """
        Оценка качества извлеченных ключевых слов
        
        Args:
            predicted_keywords: Предсказанные ключевые слова
            target_keywords: Целевые ключевые слова
            
        Returns:
            Словарь с метриками
        """
        try:
            # Приведение к множествам
            pred_set = set(predicted_keywords)
            target_set = set(target_keywords)
            
            # Расчет пересечения
            intersection = pred_set.intersection(target_set)
            
            # Расчет метрик
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(target_set) if target_set else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'keyword_precision': precision,
                'keyword_recall': recall,
                'keyword_f1': f1,
                'num_predicted': len(pred_set),
                'num_target': len(target_set),
                'num_matched': len(intersection)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при оценке ключевых слов: {e}")
            raise
            
    def calculate_confidence_metrics(
        self,
        confidences: torch.Tensor,
        correct_predictions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Расчет метрик уверенности модели
        
        Args:
            confidences: Оценки уверенности модели
            correct_predictions: Индикаторы правильных предсказаний
            
        Returns:
            Словарь с метриками уверенности
        """
        try:
            confidences = confidences.cpu().numpy()
            correct_predictions = correct_predictions.cpu().numpy()
            
            # Средняя уверенность
            mean_confidence = np.mean(confidences)
            
            # Средняя уверенность для правильных и неправильных предсказаний
            correct_conf = np.mean(confidences[correct_predictions])
            incorrect_conf = np.mean(confidences[~correct_predictions])
            
            # Разница между уверенностью для правильных и неправильных предсказаний
            confidence_gap = correct_conf - incorrect_conf
            
            return {
                'mean_confidence': float(mean_confidence),
                'correct_confidence': float(correct_conf),
                'incorrect_confidence': float(incorrect_conf),
                'confidence_gap': float(confidence_gap)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при расчете метрик уверенности: {e}")
            raise
