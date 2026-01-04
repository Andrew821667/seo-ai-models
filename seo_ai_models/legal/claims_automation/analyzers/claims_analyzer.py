"""
Анализатор претензионно-исковой работы.
Оценивает перспективы дел и предоставляет рекомендации.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from seo_ai_models.legal.claims_automation.models.claims_models import (
    Claim, CourtClaim, ClaimAnalysis, LimitationPeriod, LimitationPeriodType
)
from seo_ai_models.legal.court_practice.analyzers.court_case_analyzer import CourtCaseAnalyzer
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClaimsAnalyzer:
    """
    Анализатор претензий и исковых заявлений.
    """

    def __init__(self):
        """Инициализация анализатора."""
        self.court_analyzer = CourtCaseAnalyzer()
        self.text_processor = EnhancedTextProcessor()

        # Статистические данные по типам претензий
        self.claim_success_rates = {
            'contract_breach': 0.75,    # 75% удовлетворения
            'payment_delay': 0.85,      # 85% удовлетворения
            'quality_defect': 0.60,     # 60% удовлетворения
            'delivery_delay': 0.70,     # 70% удовлетворения
            'property_damage': 0.65,    # 65% удовлетворения
            'service_defect': 0.55,     # 55% удовлетворения
            'other': 0.50              # 50% для прочих
        }

        # Стоимость судебного разбирательства (примерные данные)
        self.court_costs = {
            'up_to_100k': 15000,        # до 100 тыс. руб.
            '100k_to_500k': 30000,      # 100-500 тыс. руб.
            '500k_to_1m': 50000,        # 500 тыс. - 1 млн. руб.
            'over_1m': 100000          # свыше 1 млн. руб.
        }

    def analyze_claim(self, claim: Claim) -> ClaimAnalysis:
        """
        Комплексный анализ претензии.

        Args:
            claim: Претензия для анализа

        Returns:
            ClaimAnalysis: Результаты анализа
        """
        try:
            # Юридический анализ
            legal_analysis = self._perform_legal_analysis(claim)

            # Оценка рисков
            risk_assessment = self._assess_claim_risks(claim)

            # Предсказание успеха
            success_prediction = self._predict_claim_success(claim)

            # Поиск похожих дел
            similar_cases = self._find_similar_claims(claim)

            # Рекомендации
            recommendations = self._generate_claim_recommendations(claim, success_prediction, risk_assessment)

            return ClaimAnalysis(
                claim=claim,
                legal_analysis=legal_analysis,
                risk_assessment=risk_assessment,
                success_prediction=success_prediction,
                similar_cases=similar_cases,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error analyzing claim {claim.claim_id}: {str(e)}")
            return ClaimAnalysis(
                claim=claim,
                recommendations=["Ошибка анализа - требуется ручная проверка"]
            )

    def analyze_court_claim(self, court_claim: CourtClaim) -> ClaimAnalysis:
        """
        Комплексный анализ искового заявления.

        Args:
            court_claim: Исковое заявление для анализа

        Returns:
            ClaimAnalysis: Результаты анализа
        """
        try:
            # Юридический анализ
            legal_analysis = self._perform_court_legal_analysis(court_claim)

            # Оценка рисков
            risk_assessment = self._assess_court_claim_risks(court_claim)

            # Предсказание успеха
            success_prediction = self._predict_court_claim_success(court_claim)

            # Анализ сроков исковой давности
            limitation_analysis = self._analyze_limitation_period(court_claim)

            # Поиск похожих дел
            similar_cases = self._find_similar_court_cases(court_claim)

            # Рекомендации
            recommendations = self._generate_court_claim_recommendations(
                court_claim, success_prediction, risk_assessment, limitation_analysis
            )

            return ClaimAnalysis(
                claim=court_claim,
                legal_analysis=legal_analysis,
                risk_assessment=risk_assessment,
                success_prediction=success_prediction,
                similar_cases=similar_cases,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error analyzing court claim {court_claim.claim_id}: {str(e)}")
            return ClaimAnalysis(
                claim=court_claim,
                recommendations=["Ошибка анализа - требуется ручная проверка"]
            )

    def analyze_limitation_period(self, court_claim: CourtClaim) -> LimitationPeriod:
        """
        Анализ срока исковой давности.

        Args:
            court_claim: Исковое заявление

        Returns:
            LimitationPeriod: Анализ срока исковой давности
        """
        return self._analyze_limitation_period(court_claim)

    def _perform_legal_analysis(self, claim: Claim) -> Dict[str, Any]:
        """
        Юридический анализ претензии.
        """
        try:
            analysis = {
                'claim_type_analysis': {},
                'amount_analysis': {},
                'evidence_analysis': {},
                'legal_basis': []
            }

            # Анализ типа претензии
            claim_type_info = self._analyze_claim_type(claim.type)
            analysis['claim_type_analysis'] = claim_type_info

            # Анализ суммы
            if claim.amount:
                amount_category = self._categorize_claim_amount(claim.amount)
                analysis['amount_analysis'] = {
                    'category': amount_category,
                    'court_fee': self._calculate_court_fee(claim.amount),
                    'recommended_actions': self._get_amount_based_recommendations(claim.amount)
                }

            # Анализ доказательной базы
            evidence_score = self._analyze_evidence_strength(claim.description)
            analysis['evidence_analysis'] = {
                'strength_score': evidence_score,
                'missing_evidence': self._identify_missing_evidence(claim)
            }

            # Определение правовых оснований
            analysis['legal_basis'] = self._identify_legal_basis(claim)

            return analysis

        except Exception as e:
            logger.warning(f"Error in legal analysis: {str(e)}")
            return {'error': str(e)}

    def _perform_court_legal_analysis(self, court_claim: CourtClaim) -> Dict[str, Any]:
        """
        Юридический анализ искового заявления.
        """
        try:
            analysis = self._perform_legal_analysis(court_claim)  # базовый анализ

            # Дополнительный анализ для суда
            analysis.update({
                'jurisdiction_analysis': self._analyze_jurisdiction(court_claim),
                'limitation_period_analysis': self._analyze_limitation_period(court_claim).__dict__,
                'court_strategy': self._develop_court_strategy(court_claim)
            })

            return analysis

        except Exception as e:
            logger.warning(f"Error in court legal analysis: {str(e)}")
            return {'error': str(e)}

    def _analyze_claim_type(self, claim_type: str) -> Dict[str, Any]:
        """
        Анализ типа претензии.
        """
        type_mapping = {
            'contract_breach': {
                'description': 'Нарушение условий договора',
                'success_rate': 0.75,
                'typical_duration': 60,  # дней
                'required_evidence': ['договор', 'акты', 'переписка']
            },
            'payment_delay': {
                'description': 'Просрочка платежа',
                'success_rate': 0.85,
                'typical_duration': 45,
                'required_evidence': ['договор', 'счета', 'уведомления']
            },
            'quality_defect': {
                'description': 'Дефекты качества товара/услуги',
                'success_rate': 0.60,
                'typical_duration': 90,
                'required_evidence': ['договор', 'акты приемки', 'экспертиза']
            },
            'delivery_delay': {
                'description': 'Просрочка поставки',
                'success_rate': 0.70,
                'typical_duration': 75,
                'required_evidence': ['договор', 'накладные', 'переписка']
            }
        }

        return type_mapping.get(str(claim_type), {
            'description': 'Прочие претензии',
            'success_rate': 0.50,
            'typical_duration': 60,
            'required_evidence': ['документы', 'переписка']
        })

    def _analyze_evidence_strength(self, description: str) -> float:
        """
        Оценка силы доказательств на основе описания.
        """
        try:
            # Анализ наличия ключевых элементов доказывания
            evidence_indicators = [
                'договор', 'акт', 'письмо', 'уведомление', 'счет', 'накладная',
                'экспертиза', 'свидетель', 'фото', 'видео', 'аудио'
            ]

            text_lower = description.lower()
            found_indicators = sum(1 for indicator in evidence_indicators if indicator in text_lower)

            # Базовый скор + бонус за найденные индикаторы
            base_score = 0.3  # минимальный скор
            indicator_bonus = min(found_indicators * 0.1, 0.5)  # до 50% за индикаторы

            return min(base_score + indicator_bonus, 1.0)

        except Exception:
            return 0.3

    def _identify_missing_evidence(self, claim: Claim) -> List[str]:
        """
        Определение недостающих доказательств.
        """
        missing = []

        # Базовые требования для всех претензий
        if 'договор' not in claim.description.lower():
            missing.append('Договор/контракт')

        # Специфические требования по типам
        if claim.type.value == 'payment_delay':
            if 'уведомление' not in claim.description.lower():
                missing.append('Уведомление о просрочке')
        elif claim.type.value == 'quality_defect':
            if 'экспертиз' not in claim.description.lower():
                missing.append('Результаты экспертизы')

        return missing

    def _identify_legal_basis(self, claim: Claim) -> List[str]:
        """
        Определение правовых оснований.
        """
        basis = []

        if claim.type.value == 'contract_breach':
            basis.extend(['ст. 309 ГК РФ', 'ст. 310 ГК РФ', 'ст. 393 ГК РФ'])
        elif claim.type.value == 'payment_delay':
            basis.extend(['ст. 395 ГК РФ', 'ст. 330 ГК РФ'])
        elif claim.type.value == 'quality_defect':
            basis.extend(['ст. 475 ГК РФ', 'ст. 503 ГК РФ'])

        return basis

    def _analyze_jurisdiction(self, court_claim: CourtClaim) -> Dict[str, Any]:
        """
        Анализ подсудности.
        """
        # Упрощенная логика определения подсудности
        analysis = {
            'recommended_court': 'Арбитражный суд',
            'jurisdiction_type': 'economic',
            'confidence': 0.8
        }

        # Определяем тип суда на основе суммы и типа спора
        if court_claim.claim_amount and court_claim.claim_amount < 50000:
            analysis.update({
                'recommended_court': 'Мировой суд',
                'jurisdiction_type': 'civil'
            })

        return analysis

    def _analyze_limitation_period(self, court_claim: CourtClaim) -> LimitationPeriod:
        """
        Анализ срока исковой давности.
        """
        try:
            # Определяем тип срока
            period_type = court_claim.limitation_period_type

            # Определяем продолжительность
            duration_mapping = {
                LimitationPeriodType.GENERAL: 1095,      # 3 года
                LimitationPeriodType.CONTRACT: 1095,     # 3 года
                LimitationPeriodType.PROPERTY: 1095,     # 3 года
                LimitationPeriodType.TORT: 1095,         # 3 года
                LimitationPeriodType.ADMINISTRATIVE: 90, # 3 месяца
                LimitationPeriodType.SHORT: 365,         # 1 год
                LimitationPeriodType.SPECIAL: 730        # 2 года (пример)
            }

            duration_days = duration_mapping.get(period_type, 1095)

            # Расчет сроков
            start_date = court_claim.created_at
            deadline = start_date + timedelta(days=duration_days)

            # Текущий статус
            now = datetime.now()
            remaining_days = (deadline - now).days
            is_expired = remaining_days <= 0

            return LimitationPeriod(
                period_type=period_type,
                start_date=start_date,
                duration_days=duration_days,
                deadline=deadline,
                remaining_days=max(0, remaining_days),
                is_expired=is_expired
            )

        except Exception as e:
            logger.error(f"Error analyzing limitation period: {str(e)}")
            # Возвращаем базовый период в случае ошибки
            return LimitationPeriod(
                period_type=LimitationPeriodType.GENERAL,
                start_date=datetime.now(),
                duration_days=1095,
                deadline=datetime.now() + timedelta(days=1095),
                remaining_days=1095,
                is_expired=False
            )

    def _assess_claim_risks(self, claim: Claim) -> Dict[str, Any]:
        """
        Оценка рисков претензии.
        """
        try:
            risk_score = 0.0
            risk_factors = []

            # Риск на основе суммы
            if claim.amount:
                if claim.amount > 1000000:
                    risk_score += 0.3
                    risk_factors.append('Высокая сумма претензии')
                elif claim.amount < 50000:
                    risk_score -= 0.1  # низкий риск для малых сумм

            # Риск на основе типа претензии
            success_rate = self.claim_success_rates.get(str(claim.type), 0.5)
            risk_score += (1 - success_rate) * 0.4

            # Риск на основе доказательной базы
            evidence_strength = self._analyze_evidence_strength(claim.description)
            risk_score += (1 - evidence_strength) * 0.3

            risk_level = 'low'
            if risk_score > 0.6:
                risk_level = 'high'
            elif risk_score > 0.3:
                risk_level = 'medium'

            return {
                'risk_level': risk_level,
                'risk_score': min(risk_score, 1.0),
                'risk_factors': risk_factors,
                'confidence': 0.75
            }

        except Exception as e:
            logger.warning(f"Error assessing claim risks: {str(e)}")
            return {
                'risk_level': 'medium',
                'risk_score': 0.5,
                'risk_factors': ['Ошибка оценки рисков'],
                'confidence': 0.0
            }

    def _assess_court_claim_risks(self, court_claim: CourtClaim) -> Dict[str, Any]:
        """
        Оценка рисков искового заявления.
        """
        base_risks = self._assess_claim_risks(court_claim)

        # Дополнительные риски для суда
        additional_risks = []

        # Риск пропуска срока исковой давности
        limitation = self._analyze_limitation_period(court_claim)
        if limitation.remaining_days < 30:
            base_risks['risk_score'] += 0.2
            additional_risks.append('Критически мало времени до истечения срока давности')

        # Риск неподсудности
        jurisdiction = self._analyze_jurisdiction(court_claim)
        if jurisdiction['confidence'] < 0.7:
            base_risks['risk_score'] += 0.1
            additional_risks.append('Возможные проблемы с подсудностью')

        base_risks['risk_factors'].extend(additional_risks)
        base_risks['risk_score'] = min(base_risks['risk_score'], 1.0)

        return base_risks

    def _predict_claim_success(self, claim: Claim) -> Dict[str, Any]:
        """
        Предсказание успеха претензии.
        """
        try:
            # Базовая вероятность успеха
            base_success = self.claim_success_rates.get(str(claim.type), 0.5)

            # Корректировка на основе доказательств
            evidence_bonus = self._analyze_evidence_strength(claim.description) * 0.2

            # Корректировка на основе суммы (большие суммы сложнее взыскать)
            amount_penalty = 0.0
            if claim.amount and claim.amount > 500000:
                amount_penalty = min((claim.amount - 500000) / 1000000 * 0.1, 0.2)

            final_success_rate = max(0.0, min(1.0, base_success + evidence_bonus - amount_penalty))

            return {
                'predicted_success_rate': final_success_rate,
                'confidence': 0.7,
                'factors': {
                    'base_rate': base_success,
                    'evidence_bonus': evidence_bonus,
                    'amount_penalty': amount_penalty
                }
            }

        except Exception as e:
            logger.warning(f"Error predicting claim success: {str(e)}")
            return {
                'predicted_success_rate': 0.5,
                'confidence': 0.0,
                'factors': {}
            }

    def _predict_court_claim_success(self, court_claim: CourtClaim) -> Dict[str, Any]:
        """
        Предсказание успеха искового заявления.
        """
        base_prediction = self._predict_claim_success(court_claim)

        # Дополнительные факторы для суда
        court_factors = {
            'limitation_risk': 0.0,
            'jurisdiction_risk': 0.0,
            'counterclaim_risk': 0.05  # базовый риск встречного иска
        }

        # Проверка срока давности
        limitation = self._analyze_limitation_period(court_claim)
        if limitation.remaining_days < 60:
            court_factors['limitation_risk'] = 0.15

        base_prediction['predicted_success_rate'] -= sum(court_factors.values())
        base_prediction['predicted_success_rate'] = max(0.0, base_prediction['predicted_success_rate'])
        base_prediction['court_factors'] = court_factors

        return base_prediction

    def _find_similar_claims(self, claim: Claim) -> List[Dict[str, Any]]:
        """
        Поиск похожих претензий.
        """
        # В реальной реализации здесь был бы поиск в базе данных
        return [
            {
                'case_number': 'A40-12345/2023',
                'similarity_score': 0.85,
                'outcome': 'удовлетворена',
                'amount': claim.amount * 0.9 if claim.amount else None
            }
        ]

    def _find_similar_court_cases(self, court_claim: CourtClaim) -> List[Dict[str, Any]]:
        """
        Поиск похожих судебных дел.
        """
        return self._find_similar_claims(court_claim)

    def _generate_claim_recommendations(self, claim: Claim, success_prediction: Dict[str, Any],
                                      risk_assessment: Dict[str, Any]) -> List[str]:
        """
        Генерация рекомендаций для претензии.
        """
        recommendations = []

        success_rate = success_prediction.get('predicted_success_rate', 0.5)
        risk_level = risk_assessment.get('risk_level', 'medium')

        if success_rate > 0.8:
            recommendations.append("Высокие шансы на успех - направьте претензию")
        elif success_rate < 0.3:
            recommendations.append("Низкие шансы на успех - рассмотрите альтернативные способы урегулирования")
        else:
            recommendations.append("Средние шансы - дополните доказательственную базу")

        if risk_level == 'high':
            recommendations.append("Высокий риск - проконсультируйтесь с юристом")

        # Рекомендации по доказательствам
        missing_evidence = self._identify_missing_evidence(claim)
        if missing_evidence:
            recommendations.append(f"Соберите дополнительные доказательства: {', '.join(missing_evidence)}")

        return recommendations

    def _generate_court_claim_recommendations(self, court_claim: CourtClaim, success_prediction: Dict[str, Any],
                                            risk_assessment: Dict[str, Any], limitation: LimitationPeriod) -> List[str]:
        """
        Генерация рекомендаций для искового заявления.
        """
        recommendations = self._generate_claim_recommendations(court_claim, success_prediction, risk_assessment)

        # Дополнительные рекомендации для суда
        if limitation.remaining_days < 30:
            recommendations.append("Критически мало времени - срочно подавайте иск")
        elif limitation.is_expired:
            recommendations.append("Срок исковой давности истек - иск не может быть удовлетворен")

        # Рекомендации по подсудности
        jurisdiction = self._analyze_jurisdiction(court_claim)
        if jurisdiction['recommended_court'] != 'Арбитражный суд':
            recommendations.append(f"Рекомендуется обратиться в {jurisdiction['recommended_court']}")

        return recommendations

    def _categorize_claim_amount(self, amount: float) -> str:
        """
        Категоризация суммы претензии.
        """
        if amount < 50000:
            return 'small'
        elif amount < 500000:
            return 'medium'
        elif amount < 2000000:
            return 'large'
        else:
            return 'very_large'

    def _calculate_court_fee(self, amount: float) -> float:
        """
        Расчет госпошлины.
        """
        # Упрощенный расчет госпошлины для арбитражных судов
        if amount <= 100000:
            return amount * 0.04
        elif amount <= 200000:
            return 4000 + (amount - 100000) * 0.03
        elif amount <= 1000000:
            return 7000 + (amount - 200000) * 0.02
        elif amount <= 2000000:
            return 23000 + (amount - 1000000) * 0.01
        else:
            return 33000 + (amount - 2000000) * 0.005

    def _get_amount_based_recommendations(self, amount: float) -> List[str]:
        """
        Рекомендации на основе суммы.
        """
        if amount < 50000:
            return ["Рассмотрите мировой суд", "Можно взыскать через приказное производство"]
        elif amount > 1000000:
            return ["Обязательно привлечение юриста", "Рассмотрите процедуру медиации"]
        else:
            return ["Стандартная процедура взыскания"]

    def _develop_court_strategy(self, court_claim: CourtClaim) -> Dict[str, Any]:
        """
        Разработка судебной стратегии.
        """
        strategy = {
            'recommended_approach': 'standard',
            'tactics': [],
            'risk_mitigation': []
        }

        # Определяем подход на основе типа иска
        if court_claim.type.value == 'economic_dispute':
            strategy['recommended_approach'] = 'aggressive'
            strategy['tactics'] = ['Максимально полная доказательная база', 'Привлечение свидетелей']
        else:
            strategy['recommended_approach'] = 'balanced'
            strategy['tactics'] = ['Фокус на ключевых доказательствах']

        # Меры по снижению рисков
        if court_claim.claim_amount and court_claim.claim_amount > 500000:
            strategy['risk_mitigation'].append('Обеспечительные меры')

        return strategy
