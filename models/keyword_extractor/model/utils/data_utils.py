# model/utils/data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import json
import random
from tqdm import tqdm

from ..config.logging_config import get_logger
from ..model import KeywordProcessor

logger = get_logger(__name__)

class KeywordDataset(Dataset):
    """Датасет для обучения модели извлечения ключевых слов"""
    
    def __init__(
        self,
        texts: List[str],
        keyword_labels: List[List[int]],
        trend_labels: List[List[float]],
        processor: KeywordProcessor
    ):
        """
        Инициализация датасета
        
        Args:
            texts: Список текстов
            keyword_labels: Метки ключевых слов
            trend_labels: Метки трендов
            processor: Процессор для обработки текстов
        """
        self.texts = texts
        self.keyword_labels = keyword_labels
        self.trend_labels = trend_labels
        self.processor = processor
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        keyword_label = self.keyword_labels[idx]
        trend_label = self.trend_labels[idx]
        
        # Кодирование текста
        encoded = self.processor.encode_texts(text)
        
        # Подготовка меток
        max_length = encoded['input_ids'].size(1)
        keyword_label = self._pad_or_truncate(keyword_label, max_length)
        trend_label = self._pad_or_truncate(trend_label, max_length)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'keyword_labels': torch.tensor(keyword_label, dtype=torch.long),
            'trend_labels': torch.tensor(trend_label, dtype=torch.float)
        }
        
    def _pad_or_truncate(
        self,
        sequence: List[Union[int, float]],
        target_length: int,
        pad_value: Union[int, float] = 0
    ) -> List[Union[int, float]]:
        """Дополнение или обрезка последовательности до нужной длины"""
        if len(sequence) > target_length:
            return sequence[:target_length]
        return sequence + [pad_value] * (target_length - len(sequence))

class DataCollator:
    """Коллатор для батчей данных"""
    
    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Формирование батча
        
        Args:
            batch: Список словарей с тензорами
            
        Returns:
            Словарь с батчами тензоров
        """
        return {
            key: torch.stack([example[key] for example in batch])
            for key in batch[0].keys()
        }

def load_dataset(
    data_path: Union[str, Path],
    processor: KeywordProcessor,
    max_samples: Optional[int] = None,
    validation_split: float = 0.2,
    seed: int = 42
) -> Tuple[KeywordDataset, KeywordDataset]:
    """
    Загрузка данных и создание датасетов
    
    Args:
        data_path: Путь к файлу с данными
        processor: Процессор для обработки текстов
        max_samples: Максимальное количество примеров
        validation_split: Доля данных для валидации
        seed: Сид для воспроизводимости
        
    Returns:
        Кортеж из тренировочного и валидационного датасетов
    """
    try:
        # Загрузка данных
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ограничение количества примеров
        if max_samples is not None:
            data = data[:max_samples]
            
        # Разделение на признаки и метки
        texts = [item['text'] for item in data]
        keyword_labels = [item['keyword_labels'] for item in data]
        trend_labels = [item['trend_labels'] for item in data]
        
        # Перемешивание данных
        random.seed(seed)
        indices = list(range(len(texts)))
        random.shuffle(indices)
        
        # Разделение на обучающую и валидационную выборки
        val_size = int(len(indices) * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Создание датасетов
        train_dataset = KeywordDataset(
            [texts[i] for i in train_indices],
            [keyword_labels[i] for i in train_indices],
            [trend_labels[i] for i in train_indices],
            processor
        )
        
        val_dataset = KeywordDataset(
            [texts[i] for i in val_indices],
            [keyword_labels[i] for i in val_indices],
            [trend_labels[i] for i in val_indices],
            processor
        )
        
        logger.info(
            f"Загружено {len(train_dataset)} примеров для обучения "
            f"и {len(val_dataset)} для валидации"
        )
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise

def create_dataloaders(
    train_dataset: KeywordDataset,
    val_dataset: KeywordDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Создание загрузчиков данных
    
    Args:
        train_dataset: Тренировочный датасет
        val_dataset: Валидационный датасет
        batch_size: Размер батча
        num_workers: Количество процессов для загрузки данных
        
    Returns:
        Кортеж из тренировочного и валидационного загрузчиков
    """
    collator = DataCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader

def save_predictions(
    predictions: List[Dict],
    output_path: Union[str, Path],
    include_metadata: bool = True
):
    """
    Сохранение предсказаний модели
    
    Args:
        predictions: Список предсказаний
        output_path: Путь для сохранения
        include_metadata: Включать ли метаданные
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Добавление метаданных
    if include_metadata:
        output = {
            'metadata': {
                'timestamp': str(datetime.now()),
                'num_predictions': len(predictions)
            },
            'predictions': predictions
        }
    else:
        output = predictions
        
    # Сохранение в JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Предсказания сохранены в {output_path}")
