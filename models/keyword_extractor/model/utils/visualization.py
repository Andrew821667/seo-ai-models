# model/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

from ..config.logging_config import get_logger

logger = get_logger(__name__)

class TrainingVisualizer:
    """Визуализация процесса обучения модели"""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        Инициализация визуализатора
        
        Args:
            save_dir: Директория для сохранения графиков
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        filename: Optional[str] = None
    ) -> None:
        """
        Построение графиков обучения
        
        Args:
            history: История метрик обучения
            filename: Имя файла для сохранения
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # График функций потерь
            plt.subplot(2, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # График метрик качества
            plt.subplot(2, 2, 2)
            for metric in ['precision', 'recall', 'f1']:
                if metric in history:
                    plt.plot(history[metric], label=metric.capitalize())
            plt.title('Metrics During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # График скорости обучения
            plt.subplot(2, 2, 3)
            plt.plot(history['learning_rates'])
            plt.title('Learning Rate')
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            
            plt.tight_layout()
            
            if filename and self.save_dir:
                plt.savefig(self.save_dir / filename)
                logger.info(f"График сохранен в {self.save_dir / filename}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при построении графиков обучения: {e}")
            raise
            
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        filename: Optional[str] = None
    ) -> None:
        """
        Построение тепловой карты внимания
        
        Args:
            attention_weights: Веса внимания
            tokens: Токены
            filename: Имя файла для сохранения
        """
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd'
            )
            plt.title('Attention Weights Heatmap')
            
            if filename and self.save_dir:
                plt.savefig(self.save_dir / filename)
                logger.info(f"Тепловая карта сохранена в {self.save_dir / filename}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при построении тепловой карты: {e}")
            raise

class KeywordVisualizer:
    """Визуализация результатов извлечения ключевых слов"""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_keyword_distribution(
        self,
        keywords: List[Dict[str, Union[str, float]]],
        top_n: int = 20,
        filename: Optional[str] = None
    ) -> None:
        """
        Построение распределения ключевых слов
        
        Args:
            keywords: Список словарей с ключевыми словами и их оценками
            top_n: Количество топовых слов для отображения
            filename: Имя файла для сохранения
        """
        try:
            # Подготовка данных
            df = pd.DataFrame(keywords)
            top_keywords = df.nlargest(top_n, 'score')
            
            plt.figure(figsize=(15, 8))
            
            # Построение барплота
            sns.barplot(
                data=top_keywords,
                x='keyword',
                y='score',
                palette='viridis'
            )
            
            plt.title(f'Top {top_n} Keywords by Score')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Keywords')
            plt.ylabel('Score')
            
            plt.tight_layout()
            
            if filename and self.save_dir:
                plt.savefig(self.save_dir / filename)
                logger.info(f"График распределения сохранен в {self.save_dir / filename}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при построении распределения ключевых слов: {e}")
            raise
            
    def plot_trend_analysis(
        self,
        keywords: List[Dict[str, Union[str, float]]],
        trend_scores: List[float],
        filename: Optional[str] = None
    ) -> None:
        """
        Визуализация анализа трендов
        
        Args:
            keywords: Список ключевых слов
            trend_scores: Оценки трендов
            filename: Имя файла для сохранения
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Создание scatter plot
            plt.scatter(trend_scores, range(len(keywords)), alpha=0.6)
            
            # Добавление подписей
            plt.yticks(
                range(len(keywords)),
                [k['keyword'] for k in keywords]
            )
            
            plt.title('Keyword Trends Analysis')
            plt.xlabel('Trend Score')
            plt.grid(True)
            
            plt.tight_layout()
            
            if filename and self.save_dir:
                plt.savefig(self.save_dir / filename)
                logger.info(f"График анализа трендов сохранен в {self.save_dir / filename}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при построении анализа трендов: {e}")
            raise
            
    def create_summary_report(
        self,
        metrics: Dict[str, float],
        top_keywords: List[Dict[str, Union[str, float]]],
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Создание отчета с результатами
        
        Args:
            metrics: Метрики модели
            top_keywords: Топ ключевых слов
            output_file: Путь для сохранения отчета
            
        Returns:
            Строка с отчетом
        """
        try:
            report = []
            
            # Заголовок
            report.append("# Keyword Extraction Report")
            report.append(f"Generated at: {datetime.now()}\n")
            
            # Метрики
            report.append("## Model Metrics")
            for name, value in metrics.items():
                report.append(f"- {name}: {value:.4f}")
            report.append("")
            
            # Топ ключевых слов
            report.append("## Top Keywords")
            for i, kw in enumerate(top_keywords, 1):
                report.append(
                    f"{i}. {kw['keyword']} (score: {kw['score']:.4f})"
                )
            
            report_text = "\n".join(report)
            
            if output_file:
                output_path = Path(output_file)
                output_path.write_text(report_text)
                logger.info(f"Отчет сохранен в {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Ошибка при создании отчета: {e}")
            raise
