# model/predict.py

import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

from .config.model_config import KeywordModelConfig
from .config.logging_config import get_logger
from .model.model import KeywordExtractorModel
from .monitoring.logger import KeywordExtractorLogger
from .monitoring.performance import PerformanceMonitor
from .utils.analysis import ErrorAnalyzer
from .utils.visualization import KeywordVisualizer

logger = get_logger(__name__)

class KeywordPredictor:
    """Класс для выполнения предсказаний"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cuda',
        threshold: float = 0.5,
        batch_size: int = 32
    ):
        """
        Инициализация предиктора
        
        Args:
            model_path: Путь к сохраненной модели
            device: Устройство для вычислений
            threshold: Порог уверенности
            batch_size: Размер батча
        """
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size
        
        # Загрузка модели
        self._load_model(model_path)
        
        # Инициализация мониторинга
        self.monitor = PerformanceMonitor()
        
        # Инициализация анализатора
        self.analyzer = ErrorAnalyzer()
        
    def _load_model(self, model_path: Union[str, Path]) -> None:
        """Загрузка модели и конфигурации"""
        model_path = Path(model_path)
        
        # Загрузка конфигурации
        config_path = model_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.config = KeywordModelConfig(**config['model'])
        else:
            logger.warning("Файл конфигурации не найден, используются параметры по умолчанию")
            self.config = KeywordModelConfig()
            
        # Создание и загрузка модели
        self.model = KeywordExtractorModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Модель загружена из {model_path}")
        
    def predict_text(
        self,
        text: str
    ) -> Dict[str, Union[List[Dict[str, Union[str, float]]], float]]:
        """
        Предсказание для одного текста
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с предсказаниями и метриками
        """
        try:
            # Замер времени
            start_time = self.monitor.start_batch()
            
            # Кодирование текста
            inputs = self.model.processor.encode_texts(text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
            # Обработка результатов
            keyword_probs = torch.softmax(outputs['keyword_logits'], dim=-1)
            keyword_mask = keyword_probs[:, :, 1] > self.threshold
            
            # Получение токенов и их оценок
            selected_tokens = inputs['input_ids'][keyword_mask]
            selected_scores = outputs['trend_scores'][keyword_mask]
            
            # Декодирование в слова
            keywords = self.model.processor.decode_keywords(
                selected_tokens,
                selected_scores
            )
            
            # Замер производительности
            performance = self.monitor.end_batch(start_time, 1)
            
            return {
                'keywords': keywords,
                'processing_time': performance['batch_processing_time']
            }
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise
            
    def predict_batch(
        self,
        texts: List[str]
    ) -> List[Dict[str, Union[List[Dict[str, Union[str, float]]], float]]]:
        """
        Предсказания для батча текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Список предсказаний
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            for text in batch_texts:
                prediction = self.predict_text(text)
                results.append(prediction)
        return results

def run_predictions(
    model_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    batch_size: int = 32,
    device: str = 'cuda',
    threshold: float = 0.5
) -> None:
    """
    Запуск предсказаний
    
    Args:
        model_path: Путь к модели
        input_path: Путь к входным данным
        output_path: Путь для сохранения результатов
        batch_size: Размер батча
        device: Устройство для вычислений
        threshold: Порог уверенности
    """
    try:
        # Настройка логирования
        output_dir = Path(output_path).parent
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_logger = KeywordExtractorLogger(
            name="predict",
            log_dir=log_dir
        )
        logger = file_logger.get_logger()
        
        # Инициализация предиктора
        predictor = KeywordPredictor(
            model_path=model_path,
            device=device,
            threshold=threshold,
            batch_size=batch_size
        )
        
        # Загрузка входных данных
        input_path = Path(input_path)
        if input_path.is_file():
            with open(input_path) as f:
                if input_path.suffix == '.json':
                    data = json.load(f)
                    texts = [d['text'] for d in data]
                else:
                    texts = f.readlines()
        else:
            texts = []
            for file_path in input_path.glob('*.txt'):
                with open(file_path) as f:
                    texts.append(f.read())
                    
        # Выполнение предсказаний
        logger.info(f"Обработка {len(texts)} текстов...")
        results = predictor.predict_batch(texts)
        
        # Подготовка результатов
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'threshold': threshold,
                'num_texts': len(texts)
            },
            'predictions': results
        }
        
        # Сохранение результатов
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Результаты сохранены в {output_path}")
        
        # Создание визуализаций
        visualizer = KeywordVisualizer(save_dir=output_dir / 'visualizations')
        
        # Распределение ключевых слов
        all_keywords = [
            kw for result in results
            for kw in result['keywords']
        ]
        visualizer.plot_keyword_distribution(
            all_keywords,
            filename='keyword_distribution.png'
        )
        
        # Создание отчета
        visualizer.create_summary_report(
            {
                'total_texts': len(texts),
                'total_keywords': len(all_keywords),
                'avg_keywords_per_text': len(all_keywords) / len(texts),
                'avg_processing_time': np.mean([r['processing_time'] for r in results])
            },
            all_keywords[:10],
            output_file=output_dir / 'prediction_report.md'
        )
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказаний: {e}")
        raise

if __name__ == '__main__':
    # Пример использования
    run_predictions(
        model_path='models/keyword_extractor.pt',
        input_path='data/test',
        output_path='outputs/predictions.json'
    )
