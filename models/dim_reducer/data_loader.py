import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from pathlib import Path
from .features import SEOFeaturesExtractor

logger = logging.getLogger(__name__)

class SEODataset(Dataset):
    """Датасет для работы с SEO данными"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        feature_extractor: SEOFeaturesExtractor,
        max_features: int = 100,
        cache_features: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.data_path = Path(data_path)
        self.feature_extractor = feature_extractor
        self.max_features = max_features
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Загрузка данных
        self.data = self._load_data()
        
        # Кэш для признаков
        self.features_cache = {}
        
    def _load_data(self) -> pd.DataFrame:
        """Загрузка данных из файла"""
        try:
            if self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            elif self.data_path.suffix in ['.xls', '.xlsx']:
                data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            logger.info(f"Loaded {len(data)} samples from {self.data_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def _extract_features(self, idx: int) -> torch.Tensor:
        """Извлечение признаков для одного семпла"""
        try:
            # Проверяем кэш
            if self.cache_features and idx in self.features_cache:
                return self.features_cache[idx]
            
            # Получаем текст и HTML если есть
            row = self.data.iloc[idx]
            text = row['text'] if 'text' in row else ''
            html = row['html'] if 'html' in row else None
            
            # Извлекаем признаки
            features = self.feature_extractor.extract_all_features(text, html)
            
            # Преобразуем в тензор
            feature_tensor = self._features_to_tensor(features)
            
            # Кэшируем если нужно
            if self.cache_features:
                self.features_cache[idx] = feature_tensor
            
            return feature_tensor
            
        except Exception as e:
            logger.error(f"Error extracting features for index {idx}: {e}")
            raise
            
    def _features_to_tensor(self, features: Dict) -> torch.Tensor:
        """Преобразование признаков в тензор"""
        # Извлекаем числовые признаки
        numeric_features = [
            features.get('word_count', 0),
            features.get('sentence_count', 0),
            features.get('avg_word_length', 0),
            features.get('vocabulary_richness', 0),
            features.get('h1_count', 0),
            features.get('h2_count', 0),
            features.get('internal_links', 0),
            features.get('external_links', 0),
            features.get('img_count', 0),
            features.get('img_alt_ratio', 0)
        ]
        
        # Добавляем TF-IDF признаки
        tfidf_features = features.get('tfidf_features', np.zeros(self.max_features))
        if len(tfidf_features) > self.max_features:
            tfidf_features = tfidf_features[:self.max_features]
        elif len(tfidf_features) < self.max_features:
            tfidf_features = np.pad(
                tfidf_features,
                (0, self.max_features - len(tfidf_features))
            )
        
        # Объединяем все признаки
        all_features = np.concatenate([numeric_features, tfidf_features])
        return torch.FloatTensor(all_features)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._extract_features(idx)
    
    def save_cache(self, path: Optional[str] = None):
        """Сохранение кэша признаков"""
        if not self.cache_features:
            return
            
        if path is None and self.cache_dir:
            path = self.cache_dir / 'features_cache.pt'
        
        if path:
            torch.save(self.features_cache, path)
            logger.info(f"Cache saved to {path}")
    
    def load_cache(self, path: Optional[str] = None):
        """Загрузка кэша признаков"""
        if not self.cache_features:
            return
            
        if path is None and self.cache_dir:
            path = self.cache_dir / 'features_cache.pt'
        
        if path and Path(path).exists():
            self.features_cache = torch.load(path)
            logger.info(f"Cache loaded from {path}")

class SEODataLoader:
    """Класс для загрузки и подготовки SEO данных"""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        max_features: int = 100,
        cache_dir: Optional[str] = None
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_features = max_features
        self.cache_dir = cache_dir
        
        self.feature_extractor = SEOFeaturesExtractor(max_features=max_features)
        
    def create_datasets(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Создание датасетов для обучения, валидации и тестирования"""
        train_dataset = SEODataset(
            train_path,
            self.feature_extractor,
            self.max_features,
            cache_dir=self.cache_dir
        )
        
        val_dataset = None
        if val_path:
            val_dataset = SEODataset(
                val_path,
                self.feature_extractor,
                self.max_features,
                cache_dir=self.cache_dir
            )
            
        test_dataset = None
        if test_path:
            test_dataset = SEODataset(
                test_path,
                self.feature_extractor,
                self.max_features,
                cache_dir=self.cache_dir
            )
            
        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Создание загрузчиков данных"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
        return train_loader, val_loader, test_loader
