"""
Обучение модели E-E-A-T на гибридном наборе данных,
который сочетает синтетические и реальные данные
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union, Tuple
import time

# Импортируем генератор синтетических данных
sys.path.append('/content/seo-ai-models')
from data_generator import generate_realistic_dataset

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridEEATTrainer:
    """Класс для обучения модели E-E-A-T на гибридных данных"""
    
    def __init__(
        self,
        output_dir: str = "hybrid_models",
        synthetic_ratio: float = 0.7,
        synthetic_size: int = 5000
    ):
        """
        Инициализация тренера
        
        Args:
            output_dir: Директория для сохранения моделей
            synthetic_ratio: Доля синтетических данных в общем наборе
            synthetic_size: Размер синтетического набора данных
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.synthetic_ratio = synthetic_ratio
        self.synthetic_size = synthetic_size
        
        self.model_types = ['random_forest', 'gradient_boosting']
    
    def load_real_data(self, real_data_path: str) -> List[Dict]:
        """
        Загрузка реальных размеченных данных
        
        Args:
            real_data_path: Путь к файлу с реальными данными (JSON или CSV)
            
        Returns:
            Список словарей с данными
        """
        path = Path(real_data_path)
        
        if not path.exists():
            logger.error(f"Файл данных не найден: {real_data_path}")
            return []
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Загружено {len(data)} записей из JSON файла")
                    return data
            elif path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
                data = df.to_dict('records')
                logger.info(f"Загружено {len(data)} записей из CSV файла")
                return data
            else:
                logger.error(f"Неподдерживаемый формат файла: {path.suffix}")
                return []
        except Exception as e:
            logger.error(f"Ошибка при загрузке реальных данных: {e}")
            return []
    
    def prepare_hybrid_dataset(self, real_data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка гибридного набора данных, объединяющего реальные и синтетические данные
        
        Args:
            real_data: Список словарей с реальными данными
            
        Returns:
            Кортеж (features, target)
        """
        # Генерация синтетических данных
        logger.info(f"Генерация {self.synthetic_size} синтетических записей")
        synthetic_data = generate_realistic_dataset(self.synthetic_size)
        
        # Подготовка реальных данных
        real_df = pd.DataFrame(real_data)
        logger.info(f"Подготовка {len(real_data)} реальных записей")
        
        # Преобразование YMYL-статуса в числовое значение
        if 'is_ymyl' in real_df.columns:
            real_df['ymyl_status'] = real_df['is_ymyl'].astype(int)
        
        # Проверка обязательных полей в реальных данных
        required_fields = [
            'expertise_score', 'authority_score', 'trust_score',
            'structural_score', 'semantic_coherence_score', 
            'overall_eeat_score'
        ]
        
        for field in required_fields:
            if field not in real_df.columns:
                logger.warning(f"Поле {field} отсутствует в реальных данных")
                # Используем значения по умолчанию для отсутствующих полей
                real_df[field] = 0.5
        
        # Дополнительно проверяем и заполняем поля, которые могут отсутствовать
        optional_fields = ['citation_score', 'external_links_score']
        for field in optional_fields:
            if field not in real_df.columns:
                logger.warning(f"Поле {field} отсутствует в реальных данных, заполняем значениями по умолчанию")
                real_df[field] = 0.0
        
        if 'ymyl_status' not in real_df.columns:
            logger.warning("Поле ymyl_status отсутствует в реальных данных, определяем по категории")
            ymyl_categories = ['finance', 'health', 'legal', 'medical', 'insurance']
            if 'category' in real_df.columns:
                real_df['ymyl_status'] = real_df['category'].apply(
                    lambda x: 1 if x in ymyl_categories else 0
                )
            else:
                real_df['ymyl_status'] = 0
        
        # Подготовка синтетических данных
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Определяем, сколько данных взять из каждого источника
        real_count = len(real_df)
        synthetic_count = len(synthetic_df)
        
        total_size = real_count + synthetic_count
        target_synthetic = int(total_size * self.synthetic_ratio)
        target_real = total_size - target_synthetic
        
        # Выборка из реальных данных
        if real_count > target_real:
            real_sample = real_df.sample(target_real, random_state=42)
            logger.info(f"Используем {target_real} из {real_count} реальных записей")
        else:
            real_sample = real_df
            logger.info(f"Используем все {real_count} реальных записей")
        
        # Выборка из синтетических данных
        if synthetic_count > target_synthetic:
            synthetic_sample = synthetic_df.sample(target_synthetic, random_state=42)
            logger.info(f"Используем {target_synthetic} из {synthetic_count} синтетических записей")
        else:
            synthetic_sample = synthetic_df
            logger.info(f"Используем все {synthetic_count} синтетических записей")
        
        # Объединение данных
        feature_columns = [
            'expertise_score', 'authority_score', 'trust_score',
            'structural_score', 'semantic_coherence_score',
            'citation_score', 'external_links_score',
            'ymyl_status'
        ]
        
        # Подготовка реальных и синтетических данных
        real_features = real_sample[feature_columns]
        real_target = real_sample['overall_eeat_score']
        
        synthetic_features = synthetic_sample[feature_columns]
        synthetic_target = synthetic_sample['overall_eeat_score']
        
        # Объединение данных
        X = pd.concat([real_features, synthetic_features], ignore_index=True)
        y = pd.concat([real_target, synthetic_target], ignore_index=True)
        
        logger.info(f"Подготовлен гибридный набор данных: {len(X)} записей ({len(real_sample)} реальных, {len(synthetic_sample)} синтетических)")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Обучение и оценка моделей для E-E-A-T
        
        Args:
            X: Признаки для обучения
            y: Целевая переменная
            
        Returns:
            Словарь с обученными моделями и результатами
        """
        # Разделение на обучающий и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Подготовка результатов
        results = {
            'trained_models': {},
            'performance': {},
            'feature_importance': {},
            'best_model': None,
            'best_score': 0
        }
        
        # Обучение разных типов моделей
        for model_type in self.model_types:
            logger.info(f"\nОбучение модели: {model_type}")
            
            # Инициализация модели
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                continue
            
            # Засекаем время обучения
            start_time = time.time()
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Замеряем время обучения
            training_time = time.time() - start_time
            
            # Прогнозирование на тестовом наборе
            y_pred = model.predict(X_test)
            
            # Оценка модели
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Важность признаков
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Сохранение результатов
            results['trained_models'][model_type] = model
            results['performance'][model_type] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'training_time': float(training_time)
            }
            results['feature_importance'][model_type] = feature_importance
            
            # Проверка, является ли эта модель лучшей
            if r2 > results['best_score']:
                results['best_model'] = model_type
                results['best_score'] = float(r2)
            
            # Вывод результатов
            logger.info(f"Модель: {model_type}")
            logger.info(f"MSE: {mse:.6f}")
            logger.info(f"RMSE: {rmse:.6f}")
            logger.info(f"R^2: {r2:.6f}")
            logger.info(f"Время обучения: {training_time:.2f} секунд")
            logger.info("\nВажность признаков:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {feature}: {importance:.4f}")
        
        return results
    
    def save_models(self, results: Dict) -> str:
        """
        Сохранение моделей и результатов
        
        Args:
            results: Словарь с результатами обучения
            
        Returns:
            Путь к сохраненной лучшей модели
        """
        # Сохраняем все модели
        for model_type, model in results['trained_models'].items():
            model_path = self.output_dir / f"eeat_hybrid_{model_type}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Модель {model_type} сохранена в {model_path}")
        
        # Сохраняем результаты в JSON
        performance_data = {
            model_type: {
                metric: float(value) if isinstance(value, np.float64) else value
                for metric, value in metrics.items()
            }
            for model_type, metrics in results['performance'].items()
        }
        
        feature_importance_data = {
            model_type: {
                feature: float(importance) if isinstance(importance, np.float64) else importance
                for feature, importance in importances.items()
            }
            for model_type, importances in results['feature_importance'].items()
        }
        
        results_data = {
            'performance': performance_data,
            'feature_importance': feature_importance_data,
            'best_model': results['best_model'],
            'best_score': float(results['best_score']) if isinstance(results['best_score'], np.float64) else results['best_score']
        }
        
        results_path = self.output_dir / "eeat_hybrid_model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Результаты сохранены в {results_path}")
        
        # Сохраняем лучшую модель отдельно для использования в EnhancedEEATAnalyzer
        best_model_path = self.output_dir / "eeat_hybrid_best_model.joblib"
        best_model = results['trained_models'][results['best_model']]
        joblib.dump(best_model, best_model_path)
        logger.info(f"Лучшая модель ({results['best_model']}) сохранена в {best_model_path}")
        
        return str(best_model_path)
    
    def visualize_results(self, X: pd.DataFrame, results: Dict) -> None:
        """
        Визуализация результатов обучения
        
        Args:
            X: Признаки, использованные для обучения
            results: Результаты обучения
        """
        # Создаем директорию для визуализаций
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Важность признаков для лучшей модели
        best_model_type = results['best_model']
        feature_importance = results['feature_importance'][best_model_type]
        
        plt.figure(figsize=(12, 6))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        indices = np.argsort(importances)
        
        plt.barh(range(len(indices)), [importances[i] for i in indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.title(f'Важность признаков ({best_model_type})')
        plt.xlabel('Относительная важность')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance_hybrid.png')
        
        # 2. Сравнение производительности моделей
        plt.figure(figsize=(10, 6))
        models = list(results['performance'].keys())
        r2_scores = [results['performance'][model]['r2'] for model in models]
        rmse_scores = [results['performance'][model]['rmse'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='skyblue')
        bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='salmon')
        
        ax1.set_xlabel('Модели')
        ax1.set_ylabel('R²', color='blue')
        ax2.set_ylabel('RMSE', color='red')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Сравнение производительности моделей')
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_performance_comparison_hybrid.png')
        
        # 3. Корреляционная матрица
        plt.figure(figsize=(12, 10))
        correlation_matrix = X.corr()
        
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(X.columns)), X.columns, rotation=45)
        plt.yticks(range(len(X.columns)), X.columns)
        
        # Добавление текстовых значений корреляции
        for i in range(len(X.columns)):
            for j in range(len(X.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black")
        
        plt.title('Корреляция между компонентами E-E-A-T (гибридные данные)')
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_matrix_hybrid.png')
        
        logger.info(f"Визуализации сохранены в директории {viz_dir}")

def main():
    """Основная функция для обучения модели на гибридных данных"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели E-E-A-T на гибридных данных')
    parser.add_argument('--real-data', type=str, help='Путь к файлу с реальными данными')
    parser.add_argument('--output-dir', type=str, default='hybrid_models', help='Директория для сохранения моделей')
    parser.add_argument('--synthetic-ratio', type=float, default=0.7, help='Доля синтетических данных')
    parser.add_argument('--synthetic-size', type=int, default=5000, help='Размер синтетического набора данных')
    
    args = parser.parse_args()
    
    trainer = HybridEEATTrainer(
        output_dir=args.output_dir,
        synthetic_ratio=args.synthetic_ratio,
        synthetic_size=args.synthetic_size
    )
    
    # Загрузка реальных данных (если указан путь)
    real_data = []
    if args.real_data:
        real_data = trainer.load_real_data(args.real_data)
    
    # Если реальных данных нет, используем только синтетические
    if not real_data:
        logger.warning("Реальные данные не найдены, используем только синтетические данные")
    
    # Подготовка гибридного набора данных
    X, y = trainer.prepare_hybrid_dataset(real_data)
    
    # Обучение моделей
    results = trainer.train_models(X, y)
    
    # Сохранение моделей
    best_model_path = trainer.save_models(results)
    
    # Визуализация результатов
    trainer.visualize_results(X, results)
    
    logger.info(f"Обучение завершено. Лучшая модель: {results['best_model']} (R² = {results['best_score']:.4f})")
    logger.info(f"Лучшая модель сохранена в {best_model_path}")

if __name__ == "__main__":
    main()
