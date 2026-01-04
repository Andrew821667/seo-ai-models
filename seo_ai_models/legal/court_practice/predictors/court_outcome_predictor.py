"""
Предиктор исходов судебных дел с использованием машинного обучения.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pickle
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from seo_ai_models.legal.court_practice.models.court_case import (
    CourtCase, CaseCategory, CaseStatus
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CourtOutcomePredictor:
    """
    Предиктор исходов судебных дел на основе машинного обучения.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация предиктора.

        Args:
            model_path: Путь к сохраненной модели
        """
        self.model_path = model_path or Path(__file__).parent / "models" / "court_outcome_model.pkl"
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []

        # Создаем директорию для моделей если не существует
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Загружаем модель если существует
        self._load_model()

        # Если модели нет, инициализируем базовую
        if self.model is None:
            self._initialize_base_model()

    def _initialize_base_model(self):
        """
        Инициализация базовой модели с правилами по умолчанию.
        """
        try:
            # Создаем простую модель на основе правил
            # В продакшене здесь должна быть полноценная ML-модель
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.scaler = StandardScaler()

            logger.info("Initialized base court outcome predictor")

        except Exception as e:
            logger.error(f"Error initializing base model: {str(e)}")

    def _load_model(self):
        """
        Загрузка обученной модели.
        """
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoders = model_data.get('label_encoders', {})
                self.feature_names = model_data.get('feature_names', [])

                logger.info(f"Loaded court outcome model from {self.model_path}")
            else:
                logger.info("No saved model found, using base model")

        except Exception as e:
            logger.warning(f"Error loading model: {str(e)}, using base model")

    def _save_model(self):
        """
        Сохранение обученной модели.
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'saved_at': datetime.now()
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved court outcome model to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def predict_outcome(self, case: CourtCase) -> Dict[str, Any]:
        """
        Предсказание исхода судебного дела.

        Args:
            case: Судебное дело для анализа

        Returns:
            Dict[str, Any]: Результаты предсказания
        """
        try:
            # Извлекаем признаки из дела
            features = self._extract_features(case)

            if not features:
                return self._fallback_prediction(case)

            # Преобразуем в numpy array
            feature_vector = np.array([list(features.values())])

            # Масштабируем признаки
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)

            # Получаем предсказание
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector)[0]

                # Маппинг классов к вероятностям
                class_probabilities = {}
                for i, class_name in enumerate(self.model.classes_):
                    class_probabilities[class_name] = float(probabilities[i])

                # Находим наиболее вероятный исход
                predicted_outcome = max(class_probabilities.items(), key=lambda x: x[1])

                return {
                    'predicted_outcome': predicted_outcome[0],
                    'confidence': predicted_outcome[1],
                    'probabilities': class_probabilities,
                    'features_used': list(features.keys()),
                    'prediction_method': 'ml_model'
                }
            else:
                # Fallback к правиловому предсказанию
                return self._fallback_prediction(case)

        except Exception as e:
            logger.error(f"Error predicting outcome for case {case.case_number}: {str(e)}")
            return self._fallback_prediction(case)

    def _extract_features(self, case: CourtCase) -> Dict[str, float]:
        """
        Извлечение признаков из судебного дела для ML-модели.
        """
        try:
            features = {}

            # Категория дела (one-hot encoding)
            for category in CaseCategory:
                features[f'category_{category.value}'] = 1.0 if case.category == category else 0.0

            # Сумма иска (логарифмированная и нормализованная)
            if case.claim_amount and case.claim_amount > 0:
                features['claim_amount_log'] = np.log(case.claim_amount)
                features['claim_amount_category'] = self._categorize_amount(case.claim_amount)
            else:
                features['claim_amount_log'] = 0.0
                features['claim_amount_category'] = 0.0

            # Количество участников
            features['plaintiffs_count'] = len(case.plaintiffs)
            features['defendants_count'] = len(case.defendants)
            features['third_parties_count'] = len(case.third_parties)
            features['total_parties'] = len(case.plaintiffs) + len(case.defendants) + len(case.third_parties)

            # Статус дела
            for status in CaseStatus:
                features[f'status_{status.value}'] = 1.0 if case.status == status else 0.0

            # Продолжительность дела (если известно)
            if case.filing_date and case.decisions:
                latest_decision = max(case.decisions, key=lambda d: d.date)
                duration = (latest_decision.date - case.filing_date).days
                features['case_duration_days'] = duration
                features['duration_category'] = self._categorize_duration(duration)
            else:
                features['case_duration_days'] = 0.0
                features['duration_category'] = 0.0

            # Количество судебных заседаний
            features['hearings_count'] = len(case.hearing_dates)

            # Количество судебных решений
            features['decisions_count'] = len(case.decisions)

            # Признаки участников (упрощенная модель)
            features['has_inn_plaintiff'] = any(p.inn for p in case.plaintiffs)
            features['has_ogrn_plaintiff'] = any(p.ogrn for p in case.plaintiffs)
            features['has_inn_defendant'] = any(p.inn for p in case.defendants)
            features['has_ogrn_defendant'] = any(p.ogrn for p in case.defendants)

            # Текстовые признаки (длина описания)
            features['description_length'] = len(case.description) if case.description else 0
            features['claim_subject_length'] = len(case.claim_subject) if case.claim_subject else 0

            # Рисковый уровень (преобразованный в числовой)
            risk_mapping = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            features['risk_level'] = risk_mapping.get(case.risk_level, 0.5)

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}

    def _categorize_amount(self, amount: float) -> float:
        """
        Категоризация суммы иска.
        """
        if amount < 100000:
            return 0.0  # маленькая
        elif amount < 1000000:
            return 0.33  # средняя
        elif amount < 10000000:
            return 0.67  # большая
        else:
            return 1.0  # очень большая

    def _categorize_duration(self, days: int) -> float:
        """
        Категоризация продолжительности дела.
        """
        if days < 30:
            return 0.0  # очень быстро
        elif days < 90:
            return 0.25  # быстро
        elif days < 180:
            return 0.5  # средне
        elif days < 365:
            return 0.75  # долго
        else:
            return 1.0  # очень долго

    def _fallback_prediction(self, case: CourtCase) -> Dict[str, Any]:
        """
        Fallback предсказание на основе правил.
        """
        try:
            # Базовые вероятности по категориям
            base_probabilities = {
                CaseCategory.CONTRACT: {'удовлетворен': 0.65, 'отклонен': 0.35},
                CaseCategory.PROPERTY: {'удовлетворен': 0.58, 'отклонен': 0.42},
                CaseCategory.CORPORATE: {'удовлетворен': 0.45, 'отклонен': 0.55},
                CaseCategory.TAX: {'удовлетворен': 0.35, 'отклонен': 0.65},
                CaseCategory.LABOR: {'удовлетворен': 0.70, 'отклонен': 0.30},
                CaseCategory.ADMINISTRATIVE: {'удовлетворен': 0.40, 'отклонен': 0.60},
                CaseCategory.OTHER: {'удовлетворен': 0.50, 'отклонен': 0.50}
            }

            probabilities = base_probabilities.get(case.category, {'удовлетворен': 0.5, 'отклонен': 0.5})

            # Корректировка на основе суммы
            if case.claim_amount:
                if case.claim_amount > 5000000:
                    probabilities['удовлетворен'] *= 0.8  # меньше шансов для больших сумм
                    probabilities['отклонен'] *= 1.2
                elif case.claim_amount < 100000:
                    probabilities['удовлетворен'] *= 1.1  # больше шансов для маленьких сумм
                    probabilities['отклонен'] *= 0.9

            # Нормализация
            total = probabilities['удовлетворен'] + probabilities['отклонен']
            probabilities['удовлетворен'] /= total
            probabilities['отклонен'] /= total

            predicted_outcome = max(probabilities.items(), key=lambda x: x[1])

            return {
                'predicted_outcome': predicted_outcome[0],
                'confidence': predicted_outcome[1],
                'probabilities': probabilities,
                'features_used': ['case_category', 'claim_amount'],
                'prediction_method': 'rule_based'
            }

        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            return {
                'predicted_outcome': 'неизвестен',
                'confidence': 0.0,
                'probabilities': {'неизвестен': 1.0},
                'features_used': [],
                'prediction_method': 'error'
            }

    def train_model(self, training_cases: List[CourtCase], outcomes: List[str]) -> Dict[str, Any]:
        """
        Обучение модели на исторических данных.

        Args:
            training_cases: Список судебных дел для обучения
            outcomes: Список исходов дел

        Returns:
            Dict[str, Any]: Результаты обучения
        """
        try:
            if len(training_cases) != len(outcomes):
                raise ValueError("Количество дел должно соответствовать количеству исходов")

            # Извлекаем признаки из всех дел
            features_list = []
            valid_cases = []
            valid_outcomes = []

            for case, outcome in zip(training_cases, outcomes):
                features = self._extract_features(case)
                if features:
                    features_list.append(features)
                    valid_cases.append(case)
                    valid_outcomes.append(outcome)

            if not features_list:
                raise ValueError("Не удалось извлечь признаки из обучающих данных")

            # Создаем DataFrame из признаков
            import pandas as pd
            df = pd.DataFrame(features_list)

            # Заполняем пропущенные значения
            df = df.fillna(0.0)

            # Кодируем целевую переменную
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(valid_outcomes)

            # Сохраняем encoder
            self.label_encoders['outcome'] = label_encoder

            # Сохраняем имена признаков
            self.feature_names = list(df.columns)

            # Разделяем на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                df.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Масштабируем признаки
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Обучаем модель
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            self.model.fit(X_train_scaled, y_train)

            # Оцениваем модель
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Сохраняем модель
            self._save_model()

            # Классификационный отчет
            report = classification_report(
                y_test, y_pred,
                target_names=label_encoder.classes_,
                output_dict=True
            )

            logger.info(f"Model trained with accuracy: {accuracy:.3f}")

            return {
                'accuracy': accuracy,
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(self.feature_names),
                'classes': list(label_encoder.classes_)
            }

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'error': str(e)}

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков модели.
        """
        try:
            if not hasattr(self.model, 'feature_importances_'):
                return {}

            importance_dict = {}
            for feature, importance in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[feature] = float(importance)

            # Сортируем по важности
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

            return sorted_importance

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
