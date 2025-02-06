# model/monitoring/logger.py

import logging
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import sys
from datetime import datetime
import os
from pythonjsonlogger import jsonlogger

class KeywordExtractorLogger:
    """Расширенный логгер для модели извлечения ключевых слов"""
    
    def __init__(
        self,
        name: str = "keyword_extractor",
        log_level: str = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        use_json: bool = False
    ):
        """
        Инициализация логгера
        
        Args:
            name: Имя логгера
            log_level: Уровень логирования
            log_dir: Директория для файлов логов
            use_json: Использовать ли JSON формат
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Очистка предыдущих обработчиков
        self.logger.handlers.clear()
        
        # Создание форматтера
        if use_json:
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_standard_formatter()
        
        # Добавление обработчика для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Добавление обработчика для файла
        if log_dir:
            self._setup_file_handler(log_dir, formatter)
            
    def _create_standard_formatter(self) -> logging.Formatter:
        """Создание стандартного форматтера"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _create_json_formatter(self) -> jsonlogger.JsonFormatter:
        """Создание JSON форматтера"""
        return jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _setup_file_handler(
        self,
        log_dir: Union[str, Path],
        formatter: Union[logging.Formatter, jsonlogger.JsonFormatter]
    ) -> None:
        """
        Настройка обработчика для файла
        
        Args:
            log_dir: Директория для логов
            formatter: Форматтер для логов
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание файла лога с текущей датой
        log_file = log_dir / f"keyword_extractor_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def get_logger(self) -> logging.Logger:
        """Получение настроенного логгера"""
        return self.logger
        
class ModelLogger:
    """Логгер для отслеживания процесса обучения и предсказаний"""
    
    def __init__(
        self,
        base_logger: logging.Logger,
        experiment_dir: Optional[Union[str, Path]] = None
    ):
        """
        Инициализация логгера модели
        
        Args:
            base_logger: Базовый логгер
            experiment_dir: Директория для сохранения результатов
        """
        self.logger = base_logger
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        if self.experiment_dir:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
        self.metrics_history = []
        
    def log_training_step(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        lr: float
    ) -> None:
        """
        Логирование шага обучения
        
        Args:
            epoch: Номер эпохи
            step: Номер шага
            metrics: Метрики
            lr: Скорость обучения
        """
        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
        )
        self.logger.info(
            f"Epoch {epoch}, Step {step} - {metrics_str} (lr: {lr:.2e})"
        )
        
        # Сохранение метрик
        self.metrics_history.append({
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Логирование результатов валидации
        
        Args:
            epoch: Номер эпохи
            metrics: Метрики валидации
        """
        metrics_str = ", ".join(
            f"val_{k}: {v:.4f}" for k, v in metrics.items()
        )
        self.logger.info(f"Validation Epoch {epoch} - {metrics_str}")
        
    def log_prediction(
        self,
        text_id: str,
        keywords: List[Dict[str, Union[str, float]]],
        processing_time: float
    ) -> None:
        """
        Логирование предсказаний
        
        Args:
            text_id: Идентификатор текста
            keywords: Предсказанные ключевые слова
            processing_time: Время обработки
        """
        self.logger.debug(
            f"Prediction for {text_id} - "
            f"Found {len(keywords)} keywords in {processing_time:.2f}s"
        )
        
    def save_metrics_history(self) -> None:
        """Сохранение истории метрик"""
        if self.experiment_dir and self.metrics_history:
            metrics_file = self.experiment_dir / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
                
            self.logger.info(f"Metrics history saved to {metrics_file}")
            
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Логирование ошибок
        
        Args:
            error: Объект ошибки
            context: Контекст ошибки
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            error_info['context'] = context
            
        self.logger.error(
            "Error occurred",
            extra={'error_details': error_info}
        )
        
    def log_system_info(self) -> None:
        """Логирование информации о системе"""
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'environment': {k: v for k, v in os.environ.items() if 'PATH' not in k},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            import torch
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_device'] = torch.cuda.get_device_name(0)
        except ImportError:
            system_info['cuda_available'] = False
            
        self.logger.info(
            "System information",
            extra={'system_info': system_info}
        )
