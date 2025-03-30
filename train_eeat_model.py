"""
Обучение модели E-E-A-T анализа на реалистичных данных
"""

import sys
import os
sys.path.append('/content/seo-ai-models')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import json

from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from data_generator import generate_realistic_dataset, analyze_dataset

# Установка параметров
DATA_SIZE = 5000  # Количество примеров
TEST_SIZE = 0.2   # Доля тестовых данных
MODEL_TYPES = ['random_forest', 'gradient_boosting']
OUTPUT_DIR = '/content/seo-ai-models/models/checkpoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_models(data, model_types):
    """
    Обучение и оценка моделей для E-E-A-T
    
    Args:
        data: Список словарей с данными
        model_types: Список типов моделей для обучения
    
    Returns:
        Словарь с обученными моделями и результатами
    """
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)
    
    # Подготовка данных для обучения
    X = df[[
        'expertise_score', 'authority_score', 'trust_score',
        'structural_score', 'semantic_coherence_score',
        'citation_score', 'external_links_score',
        'ymyl_status'  # Преобразуем булево значение в числовое
    ]].copy()
    
    # Преобразуем булевый столбец в числовой (0/1)
    X['ymyl_status'] = X['ymyl_status'].astype(int)
    
    y = df['overall_eeat_score']
    
    # Разделение на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
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
    for model_type in model_types:
        print(f"\nОбучение модели: {model_type}")
        
        # Инициализация модели
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
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
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time
        }
        results['feature_importance'][model_type] = feature_importance
        
        # Проверка, является ли эта модель лучшей
        if r2 > results['best_score']:
            results['best_model'] = model_type
            results['best_score'] = r2
        
        # Вывод результатов
        print(f"Модель: {model_type}")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R^2: {r2:.6f}")
        print(f"Время обучения: {training_time:.2f} секунд")
        print("\nВажность признаков:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
    
    return results

def save_models(results, output_dir):
    """
    Сохранение моделей и результатов
    
    Args:
        results: Словарь с результатами обучения
        output_dir: Директория для сохранения
    """
    # Создаем директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем все модели
    for model_type, model in results['trained_models'].items():
        model_path = os.path.join(output_dir, f"eeat_{model_type}.joblib")
        joblib.dump(model, model_path)
        print(f"Модель {model_type} сохранена в {model_path}")
    
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
    
    results_path = os.path.join(output_dir, "eeat_model_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Результаты сохранены в {results_path}")
    
    # Сохраняем лучшую модель отдельно для использования в EnhancedEEATAnalyzer
    best_model_path = os.path.join(output_dir, "eeat_best_model.joblib")
    best_model = results['trained_models'][results['best_model']]
    joblib.dump(best_model, best_model_path)
    print(f"Лучшая модель ({results['best_model']}) сохранена в {best_model_path}")
    
    return best_model_path

def visualize_results(data, results, output_dir):
    """
    Визуализация результатов анализа данных и обучения
    
    Args:
        data: Исходные данные
        results: Результаты обучения
        output_dir: Директория для сохранения
    """
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)
    
    # Создаем директорию для визуализаций
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Распределение общей оценки E-E-A-T
    plt.figure(figsize=(10, 6))
    plt.hist(df['overall_eeat_score'], bins=30, alpha=0.7, color='blue')
    plt.title('Распределение общей оценки E-E-A-T')
    plt.xlabel('Оценка')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'overall_score_distribution.png'))
    
    # 2. Важность признаков для лучшей модели
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
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
    
    # 3. Сравнение YMYL и не-YMYL оценок
    plt.figure(figsize=(10, 6))
    ymyl_scores = df[df['ymyl_status'] == True]['overall_eeat_score']
    non_ymyl_scores = df[df['ymyl_status'] == False]['overall_eeat_score']
    
    plt.hist(ymyl_scores, bins=20, alpha=0.5, label='YMYL', color='red')
    plt.hist(non_ymyl_scores, bins=20, alpha=0.5, label='Не-YMYL', color='blue')
    plt.title('Сравнение оценок для YMYL и не-YMYL контента')
    plt.xlabel('Оценка')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'ymyl_vs_non_ymyl.png'))
    
    # 4. Распределение по отраслям
    plt.figure(figsize=(12, 6))
    industry_means = df.groupby('industry')['overall_eeat_score'].mean().sort_values()
    
    industry_means.plot(kind='bar', color='skyblue')
    plt.title('Средняя оценка E-E-A-T по отраслям')
    plt.xlabel('Отрасль')
    plt.ylabel('Средняя оценка')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'industry_scores.png'))
    
    # 5. Корреляционная матрица
    plt.figure(figsize=(12, 10))
    corr_columns = [
        'expertise_score', 'authority_score', 'trust_score',
        'structural_score', 'semantic_coherence_score',
        'citation_score', 'external_links_score', 'overall_eeat_score'
    ]
    corr_matrix = df[corr_columns].corr()
    
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_columns)), corr_columns, rotation=45)
    plt.yticks(range(len(corr_columns)), corr_columns)
    
    # Добавление текстовых значений корреляции
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    plt.title('Корреляция между компонентами E-E-A-T')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
    
    print(f"Визуализации сохранены в директории {viz_dir}")

def test_trained_model(best_model_path):
    """
    Тестирование обученной модели с использованием EnhancedEEATAnalyzer
    
    Args:
        best_model_path: Путь к сохраненной лучшей модели
    """
    print("\nТестирование обученной модели на примере...")
    
    # Тестовый текст
    test_text = """
    # Повышение пенсий в 2023 году: новые правила и расчеты
    
    Министерство финансов РФ сообщает о планируемом повышении пенсионных выплат с 1 января 2023 года. 
    Согласно официальным данным, индексация составит 8,6%, что выше уровня инфляции за прошедший период.
    
    ## Категории получателей пенсий
    
    Повышение коснется следующих категорий:
    * Получатели страховых пенсий по старости
    * Лица, получающие пенсии по инвалидности
    * Получатели пенсий по потере кормильца
    
    По оценкам экспертов Пенсионного фонда, данное повышение затронет более 40 млн граждан России.
    
    ## Порядок расчета
    
    Для расчета новой суммы пенсии применяется следующая формула:
    
    Новая пенсия = Текущая пенсия × 1.086
    
    | Текущая пенсия (руб.) | Прибавка (руб.) | Новая пенсия (руб.) |
    |-----------------------|-----------------|---------------------|
    | 10 000                | 860             | 10 860              |
    | 15 000                | 1 290           | 16 290              |
    | 20 000                | 1 720           | 21 720              |
    
    Как отметил главный экономист Центра аналитики и финансовых технологий: «Данное повышение позволит частично компенсировать рост цен, наблюдавшийся в 2022 году, и поддержать покупательную способность пенсионеров».
    
    ## Источники финансирования
    
    Финансирование повышения будет осуществляться из федерального бюджета. По данным Министерства финансов, на эти цели выделено 172,7 млрд рублей.
    
    © 2023 ФинансИнфо. Информация актуальна на 15.02.2023
    """
    
    # Инициализация анализатора с обученной моделью
    analyzer = EnhancedEEATAnalyzer(model_path=best_model_path)
    
    # Анализ с использованием обученной модели
    result = analyzer.analyze(test_text, industry='finance')
    
    # Вывод результатов
    print(f"\nРезультаты анализа:")
    print(f"Общая оценка E-E-A-T: {result['overall_eeat_score']:.4f}")
    print(f"Экспертиза (E): {result['expertise_score']:.4f}")
    print(f"Авторитетность (A): {result['authority_score']:.4f}")
    print(f"Доверие (T): {result['trust_score']:.4f}")
    print(f"Структура: {result['structural_score']:.4f}")
    print(f"Семантическая связность: {result['semantic_coherence_score']:.4f}")
    
    # Вывод некоторых рекомендаций
    print("\nОсновные рекомендации:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    print(f"Генерация реалистичного набора данных (размер={DATA_SIZE})...")
    data = generate_realistic_dataset(DATA_SIZE)
    
    print("\nАнализ сгенерированных данных...")
    analysis = analyze_dataset(data)
    print(f"YMYL распределение: {analysis['ymyl_distribution']}")
    print(f"Средняя оценка YMYL: {analysis['ymyl_vs_non_ymyl']['ymyl_mean_score']:.4f}")
    print(f"Средняя оценка не-YMYL: {analysis['ymyl_vs_non_ymyl']['non_ymyl_mean_score']:.4f}")
    
    print("\nОбучение моделей...")
    results = train_models(data, MODEL_TYPES)
    
    print("\nСохранение моделей и результатов...")
    best_model_path = save_models(results, OUTPUT_DIR)
    
    print("\nСоздание визуализаций...")
    visualize_results(data, results, OUTPUT_DIR)
    
    # Тестирование обученной модели
    test_trained_model(best_model_path)
