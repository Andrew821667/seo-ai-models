import torch
from typing import Dict, List, Union
import numpy as np
import logging
from pathlib import Path

from common.config.dim_reducer_config import DimReducerConfig
from .model import DimensionReducer
from common.utils.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

class DimReducerInference:
    """Класс для инференса модели DimensionReducer"""
    
    def __init__(
        self,
        model_path: str,
        config: DimReducerConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Инициализация модели
        self.model = DimensionReducer(config)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Препроцессор для текста
        self.preprocessor = TextPreprocessor(config.max_length)
        
    def reduce_dimensions(
        self,
        features: Union[torch.Tensor, np.ndarray],
        return_importance: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Сжатие размерности входных признаков
        Args:
            features: входные признаки
            return_importance: возвращать ли важность признаков
        Returns:
            словарь с результатами
        """
        try:
            # Преобразование входных данных в тензор
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            features = features.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(features)
                
            results = {
                'latent_features': outputs['latent'].cpu().numpy(),
                'reconstructed': outputs['reconstructed'].cpu().numpy()
            }
            
            if return_importance:
                results['feature_importance'] = outputs['importance'].cpu().numpy()
                
            return results
            
        except Exception as e:
            logger.error(f"Error during dimension reduction: {e}")
            raise

    def process_text(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Обработка текстовых данных
        Args:
            texts: входной текст или список текстов
        Returns:
            словарь с результатами обработки
        """
        try:
            # Предобработка текста
            features = self.preprocessor.process(texts)
            
            # Сжатие размерности
            results = self.reduce_dimensions(features)
            
            # Добавление метаданных
            if isinstance(texts, str):
                texts = [texts]
            results['num_samples'] = len(texts)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during text processing: {e}")
            raise
            
    def analyze_feature_importance(
        self,
        features: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Анализ важности признаков
        Args:
            features: входные признаки
        Returns:
            словарь с результатами анализа
        """
        try:
            results = self.reduce_dimensions(features, return_importance=True)
            
            # Расчет статистик важности
            importance_scores = results['feature_importance']
            
            analysis = {
                'mean_importance': np.mean(importance_scores, axis=0),
                'std_importance': np.std(importance_scores, axis=0),
                'feature_ranking': np.argsort(-importance_scores.mean(axis=0))
            }
            
            return {**results, **analysis}
            
        except Exception as e:
            logger.error(f"Error during feature importance analysis: {e}")
            raise

    def batch_process(
        self,
        features: Union[torch.Tensor, np.ndarray],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Пакетная обработка данных
        Args:
            features: входные признаки
            batch_size: размер пакета
        Returns:
            словарь с результатами обработки
        """
        try:
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
                
            num_samples = len(features)
            all_results = []
            
            for i in range(0, num_samples, batch_size):
                batch = features[i:i + batch_size].to(self.device)
                batch_results = self.reduce_dimensions(batch)
                all_results.append(batch_results)
                
            # Объединение результатов
            combined_results = {}
            for key in all_results[0].keys():
                combined_results[key] = np.concatenate(
                    [r[key] for r in all_results],
                    axis=0
                )
                
            return combined_results
            
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            raise
