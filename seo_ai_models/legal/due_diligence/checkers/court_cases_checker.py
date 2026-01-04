"""
Чекер судебных дел контрагента для due diligence.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from seo_ai_models.legal.due_diligence.models.due_diligence import (
    DueDiligenceCheck, CheckType, RiskLevel, CourtCaseSummary
)
from seo_ai_models.legal.court_practice.parsers.arbitrage_court_parser import ArbitrageCourtParser
from seo_ai_models.legal.court_practice.models.court_case import CourtCase

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CourtCasesChecker:
    """
    Проверка судебных дел контрагента.
    """

    def __init__(self):
        """Инициализация чекера."""
        self.court_parser = ArbitrageCourtParser()

        # Критерии оценки рисков
        self.risk_criteria = {
            'max_cases_threshold': 10,  # максимум дел для низкого риска
            'high_amount_threshold': 1000000,  # 1 млн руб
            'recent_cases_months': 12,  # рассматривать дела за последний год
            'bankruptcy_keywords': ['банкротство', 'ликвидация', 'несостоятельность'],
            'fraud_keywords': ['мошенничество', 'обман', 'мошеннический']
        }

    def check_court_cases(
        self,
        company_name: str,
        inn: Optional[str] = None,
        ogrn: Optional[str] = None
    ) -> DueDiligenceCheck:
        """
        Проверка судебных дел контрагента.

        Args:
            company_name: Название компании
            inn: ИНН компании
            ogrn: ОГРН компании

        Returns:
            DueDiligenceCheck: Результат проверки
        """
        try:
            check = DueDiligenceCheck(
                check_type=CheckType.COURT_CASES,
                status="in_progress"
            )

            # Формируем поисковый запрос
            search_query = self._build_search_query(company_name, inn, ogrn)

            # Ищем судебные дела
            cases = self.court_parser.search_cases(
                query=search_query,
                max_results=100  # проверяем до 100 дел
            )

            # Анализируем найденные дела
            analysis_result = self._analyze_cases(cases, company_name)

            # Определяем уровень риска
            risk_level, score, findings, recommendations = self._assess_risk(analysis_result)

            # Формируем результат
            check.status = "completed"
            check.risk_level = risk_level
            check.score = score
            check.findings = findings
            check.recommendations = recommendations
            check.data = {
                'search_query': search_query,
                'cases_found': len(cases),
                'analysis': analysis_result,
                'cases_details': [
                    {
                        'case_number': case.case_number,
                        'court_name': case.court_name,
                        'category': case.category.value,
                        'status': case.status.value,
                        'claim_amount': case.claim_amount,
                        'filing_date': case.filing_date.isoformat() if case.filing_date else None
                    } for case in cases[:10]  # первые 10 дел
                ]
            }
            check.checked_at = datetime.now()

            logger.info(f"Court cases check completed for {company_name}: {len(cases)} cases found")
            return check

        except Exception as e:
            logger.error(f"Error checking court cases for {company_name}: {str(e)}")
            return DueDiligenceCheck(
                check_type=CheckType.COURT_CASES,
                status="failed",
                risk_level=RiskLevel.HIGH,  # в случае ошибки считаем высокий риск
                score=0.0,
                findings=["Ошибка проверки судебных дел"],
                error_message=str(e),
                checked_at=datetime.now()
            )

    def _build_search_query(self, company_name: str, inn: Optional[str], ogrn: Optional[str]) -> str:
        """
        Формирование поискового запроса для судебных дел.
        """
        query_parts = []

        # Добавляем название компании
        query_parts.append(company_name)

        # Добавляем ИНН если есть
        if inn:
            query_parts.append(f"ИНН {inn}")

        # Добавляем ОГРН если есть
        if ogrn:
            query_parts.append(f"ОГРН {ogrn}")

        return " ".join(query_parts)

    def _analyze_cases(self, cases: List[CourtCase], company_name: str) -> Dict[str, Any]:
        """
        Анализ найденных судебных дел.
        """
        if not cases:
            return {
                'total_cases': 0,
                'plaintiff_cases': 0,
                'defendant_cases': 0,
                'won_cases': 0,
                'lost_cases': 0,
                'pending_cases': 0,
                'total_amount': 0.0,
                'average_amount': 0.0,
                'recent_cases': [],
                'risk_indicators': []
            }

        # Классифицируем дела по участию компании
        plaintiff_cases = []
        defendant_cases = []

        for case in cases:
            # Определяем участие компании в деле
            is_plaintiff = self._is_company_participant(case.plaintiffs, company_name)
            is_defendant = self._is_company_participant(case.defendants, company_name)

            if is_plaintiff:
                plaintiff_cases.append(case)
            if is_defendant:
                defendant_cases.append(case)

        # Анализируем исходы дел
        won_cases = 0
        lost_cases = 0
        pending_cases = 0

        for case in plaintiff_cases + defendant_cases:
            if case.status.value == 'decided':
                # Определяем исход (упрощенная логика)
                if case.decisions:
                    last_decision = case.decisions[-1]
                    if 'удовлетвори' in (last_decision.outcome or '').lower():
                        won_cases += 1
                    elif 'отказ' in (last_decision.outcome or '').lower():
                        lost_cases += 1
            else:
                pending_cases += 1

        # Считаем суммы
        total_amount = sum(case.claim_amount for case in cases if case.claim_amount)
        average_amount = total_amount / len(cases) if cases else 0

        # Недавние дела (за последний год)
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_cases = [
            case for case in cases
            if case.filing_date and case.filing_date > one_year_ago
        ]

        # Индикаторы риска
        risk_indicators = self._identify_risk_indicators(cases, company_name)

        return {
            'total_cases': len(cases),
            'plaintiff_cases': len(plaintiff_cases),
            'defendant_cases': len(defendant_cases),
            'won_cases': won_cases,
            'lost_cases': lost_cases,
            'pending_cases': pending_cases,
            'total_amount': total_amount,
            'average_amount': average_amount,
            'recent_cases': recent_cases[:5],  # последние 5 дел
            'risk_indicators': risk_indicators
        }

    def _is_company_participant(self, participants: List, company_name: str) -> bool:
        """
        Определяет, участвует ли компания в списке участников.
        """
        company_name_lower = company_name.lower()

        for participant in participants:
            if hasattr(participant, 'name'):
                participant_name = participant.name.lower()
                # Проверяем на совпадение названия
                if company_name_lower in participant_name or participant_name in company_name_lower:
                    return True

        return False

    def _identify_risk_indicators(self, cases: List[CourtCase], company_name: str) -> List[str]:
        """
        Выявление индикаторов риска в судебных делах.
        """
        indicators = []

        # Проверяем на дела о банкротстве
        bankruptcy_cases = []
        for case in cases:
            case_text = f"{case.description} {case.claim_subject}".lower()
            if any(keyword in case_text for keyword in self.risk_criteria['bankruptcy_keywords']):
                bankruptcy_cases.append(case.case_number)

        if bankruptcy_cases:
            indicators.append(f"Дела о банкротстве/ликвидации: {len(bankruptcy_cases)} дел")

        # Проверяем на дела о мошенничестве
        fraud_cases = []
        for case in cases:
            case_text = f"{case.description} {case.claim_subject}".lower()
            if any(keyword in case_text for keyword in self.risk_criteria['fraud_keywords']):
                fraud_cases.append(case.case_number)

        if fraud_cases:
            indicators.append(f"Дела связанные с мошенничеством: {len(fraud_cases)} дел")

        # Проверяем на крупные суммы
        large_amount_cases = [
            case for case in cases
            if case.claim_amount and case.claim_amount > self.risk_criteria['high_amount_threshold']
        ]
        if large_amount_cases:
            indicators.append(f"Дела с суммой >1 млн руб: {len(large_amount_cases)} дел")

        return indicators

    def _assess_risk(self, analysis: Dict[str, Any]) -> tuple[RiskLevel, float, List[str], List[str]]:
        """
        Оценка уровня риска на основе анализа дел.
        """
        total_cases = analysis['total_cases']
        risk_indicators = analysis['risk_indicators']

        # Базовый скор (100 - идеально, 0 - критично)
        score = 100.0
        findings = []
        recommendations = []

        # Штрафуем за количество дел
        if total_cases > self.risk_criteria['max_cases_threshold']:
            penalty = min(50, (total_cases - self.risk_criteria['max_cases_threshold']) * 5)
            score -= penalty
            findings.append(f"Много судебных дел: {total_cases} (порог: {self.risk_criteria['max_cases_threshold']})")

        # Штрафуем за проигранные дела
        lost_cases = analysis['lost_cases']
        if lost_cases > 0:
            penalty = min(30, lost_cases * 10)
            score -= penalty
            findings.append(f"Проигранные дела: {lost_cases}")

        # Штрафуем за индикаторы риска
        if risk_indicators:
            penalty = len(risk_indicators) * 20
            score -= penalty
            findings.extend(risk_indicators)

        # Проверяем недавние дела
        recent_cases = analysis['recent_cases']
        if len(recent_cases) > 5:
            score -= 10
            findings.append(f"Много недавних дел: {len(recent_cases)}")

        # Определяем уровень риска
        if score >= 80:
            risk_level = RiskLevel.LOW
        elif score >= 60:
            risk_level = RiskLevel.MEDIUM
        elif score >= 40:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # Формируем рекомендации
        if total_cases == 0:
            recommendations.append("Судебных дел не найдено - положительный фактор")
        elif total_cases <= 3:
            recommendations.append("Небольшое количество дел - допустимо для активной компании")
        else:
            recommendations.append("Рекомендуется детально изучить все судебные дела")

        if risk_indicators:
            recommendations.append("Обратить внимание на дела связанные с банкротством и мошенничеством")

        if lost_cases > won_cases:
            recommendations.append("Преобладание проигранных дел - повышенное внимание к судебным рискам")

        return risk_level, max(0, score), findings, recommendations
