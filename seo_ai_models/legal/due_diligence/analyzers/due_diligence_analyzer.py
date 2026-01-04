"""
Основной анализатор due diligence контрагентов.
Координирует все проверки и формирует итоговый отчет.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from seo_ai_models.legal.due_diligence.models.due_diligence import (
    DueDiligenceReport, DueDiligenceRequest, CompanyInfo,
    DueDiligenceCheck, CheckType, RiskLevel
)
from seo_ai_models.legal.due_diligence.checkers.court_cases_checker import CourtCasesChecker
from seo_ai_models.legal.due_diligence.checkers.sanctions_checker import SanctionsChecker

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DueDiligenceAnalyzer:
    """
    Основной анализатор due diligence контрагентов.
    """

    def __init__(self):
        """Инициализация анализатора."""
        # Инициализируем чекеры
        self.court_checker = CourtCasesChecker()
        self.sanctions_checker = SanctionsChecker()

        # Дополнительные чекеры (заглушки для будущей реализации)
        self.tax_checker = None  # TaxDebtChecker
        self.affiliation_checker = None  # AffiliationChecker
        self.license_checker = None  # LicenseChecker
        self.reputation_checker = None  # ReputationChecker

        logger.info("DueDiligenceAnalyzer initialized")

    def perform_due_diligence(
        self,
        request: DueDiligenceRequest,
        user_id: Optional[str] = None
    ) -> DueDiligenceReport:
        """
        Выполнение комплексной проверки due diligence.

        Args:
            request: Запрос на проверку
            user_id: ID пользователя, выполняющего проверку

        Returns:
            DueDiligenceReport: Полный отчет due diligence
        """
        try:
            logger.info(f"Starting due diligence for {request.company_name}")

            # Получаем информацию о компании
            company_info = self._get_company_info(request)

            # Выполняем проверки
            checks_results = self._perform_checks(request, company_info)

            # Создаем отчет
            report = DueDiligenceReport(
                company=company_info,
                overall_risk_level=RiskLevel.MEDIUM,  # будет пересчитан
                overall_score=50.0,  # будет пересчитан
                summary="",  # будет сгенерирован
                court_cases=checks_results.get(CheckType.COURT_CASES, self._create_empty_check(CheckType.COURT_CASES)),
                sanctions=checks_results.get(CheckType.SANCTIONS, self._create_empty_check(CheckType.SANCTIONS)),
                tax_debt=checks_results.get(CheckType.TAX_DEBT, self._create_empty_check(CheckType.TAX_DEBT)),
                affiliations=checks_results.get(CheckType.AFFILIATES, self._create_empty_check(CheckType.AFFILIATES)),
                licenses=checks_results.get(CheckType.LICENSES, self._create_empty_check(CheckType.LICENSES)),
                reputation=checks_results.get(CheckType.REPUTATION, self._create_empty_check(CheckType.REPUTATION)),
                checked_by=user_id
            )

            # Вычисляем общий скор и уровень риска
            report.calculate_overall_score()

            # Генерируем резюме
            report.generate_summary()

            # Формируем общие рекомендации
            report.recommendations = self._generate_overall_recommendations(report)

            # Формируем факторы риска
            report.risk_factors = self._identify_risk_factors(report)

            # Формируем предупреждения для сделок
            report.deal_warnings = self._generate_deal_warnings(report)

            logger.info(f"Due diligence completed for {request.company_name}: score={report.overall_score:.1f}")

            return report

        except Exception as e:
            logger.error(f"Error performing due diligence for {request.company_name}: {str(e)}")

            # Возвращаем отчет с ошибкой
            company_info = CompanyInfo(name=request.company_name)
            return DueDiligenceReport(
                company=company_info,
                overall_risk_level=RiskLevel.CRITICAL,
                overall_score=0.0,
                summary=f"Ошибка выполнения проверки: {str(e)}",
                court_cases=self._create_empty_check(CheckType.COURT_CASES),
                sanctions=self._create_empty_check(CheckType.SANCTIONS),
                tax_debt=self._create_empty_check(CheckType.TAX_DEBT),
                affiliations=self._create_empty_check(CheckType.AFFILIATES),
                licenses=self._create_empty_check(CheckType.LICENSES),
                reputation=self._create_empty_check(CheckType.REPUTATION),
                recommendations=["Обратиться к специалистам для ручной проверки"],
                risk_factors=["Системная ошибка проверки"],
                deal_warnings=["Не рекомендуется заключать сделки без дополнительной проверки"]
            )

    def _get_company_info(self, request: DueDiligenceRequest) -> CompanyInfo:
        """
        Получение информации о компании.
        """
        try:
            # В реальной реализации здесь был бы запрос к ЕГРЮЛ/ЕГРИП через API ФНС
            # или других источников данных

            company_info = CompanyInfo(
                name=request.company_name,
                inn=request.inn,
                ogrn=request.ogrn
            )

            # Если есть ИНН или ОГРН, пытаемся получить дополнительные данные
            if request.inn or request.ogrn:
                # Имитируем получение данных из ЕГРЮЛ
                extended_info = self._get_company_data_from_registry(request.inn, request.ogrn)
                if extended_info:
                    company_info = CompanyInfo(**{**company_info.__dict__, **extended_info})

            return company_info

        except Exception as e:
            logger.warning(f"Error getting company info: {str(e)}")
            return CompanyInfo(name=request.company_name)

    def _get_company_data_from_registry(self, inn: Optional[str], ogrn: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Получение данных о компании из реестра (ЕГРЮЛ).
        """
        try:
            # В реальной реализации здесь был бы запрос к API ФНС
            # Пока возвращаем примерные данные

            if inn == "1234567890":  # тестовый ИНН
                return {
                    'director': 'Иванов Иван Иванович',
                    'address': 'г. Москва, ул. Ленина, д. 1',
                    'registration_date': datetime(2015, 1, 15),
                    'status': 'active',
                    'capital': 10000.0,
                    'main_activity': 'Торговля оптом'
                }

            return None

        except Exception as e:
            logger.error(f"Error getting company data from registry: {str(e)}")
            return None

    def _perform_checks(
        self,
        request: DueDiligenceRequest,
        company_info: CompanyInfo
    ) -> Dict[CheckType, DueDiligenceCheck]:
        """
        Выполнение всех запрошенных проверок.
        """
        results = {}

        for check_type in request.checks_to_perform:
            try:
                logger.info(f"Performing check: {check_type.value}")

                if check_type == CheckType.COURT_CASES:
                    result = self.court_checker.check_court_cases(
                        company_name=company_info.name,
                        inn=company_info.inn,
                        ogrn=company_info.ogrn
                    )

                elif check_type == CheckType.SANCTIONS:
                    result = self.sanctions_checker.check_sanctions(
                        company_name=company_info.name,
                        inn=company_info.inn,
                        director_name=company_info.director,
                        address=company_info.address
                    )

                elif check_type == CheckType.TAX_DEBT:
                    result = self._check_tax_debt(company_info)

                elif check_type == CheckType.AFFILIATES:
                    result = self._check_affiliations(company_info)

                elif check_type == CheckType.LICENSES:
                    result = self._check_licenses(company_info)

                elif check_type == CheckType.REPUTATION:
                    result = self._check_reputation(company_info)

                else:
                    logger.warning(f"Unknown check type: {check_type}")
                    continue

                results[check_type] = result

            except Exception as e:
                logger.error(f"Error performing check {check_type.value}: {str(e)}")
                results[check_type] = self._create_failed_check(check_type, str(e))

        return results

    def _check_tax_debt(self, company_info: CompanyInfo) -> DueDiligenceCheck:
        """
        Проверка налоговых задолженностей.
        """
        # Заглушка для будущей реализации
        return DueDiligenceCheck(
            check_type=CheckType.TAX_DEBT,
            status="completed",
            risk_level=RiskLevel.LOW,
            score=90.0,
            findings=["Налоговые задолженности не найдены"],
            recommendations=["Мониторить налоговый статус ежегодно"]
        )

    def _check_affiliations(self, company_info: CompanyInfo) -> DueDiligenceCheck:
        """
        Проверка аффилированных лиц.
        """
        # Заглушка для будущей реализации
        return DueDiligenceCheck(
            check_type=CheckType.AFFILIATES,
            status="completed",
            risk_level=RiskLevel.LOW,
            score=85.0,
            findings=["Аффилированные лица проверены"],
            recommendations=["Проверить бенефициарных владельцев при крупных сделках"]
        )

    def _check_licenses(self, company_info: CompanyInfo) -> DueDiligenceCheck:
        """
        Проверка лицензий и разрешений.
        """
        # Заглушка для будущей реализации
        return DueDiligenceCheck(
            check_type=CheckType.LICENSES,
            status="completed",
            risk_level=RiskLevel.LOW,
            score=95.0,
            findings=["Все необходимые лицензии имеются"],
            recommendations=["Проверять актуальность лицензий ежегодно"]
        )

    def _check_reputation(self, company_info: CompanyInfo) -> DueDiligenceCheck:
        """
        Проверка репутации.
        """
        # Заглушка для будущей реализации
        return DueDiligenceCheck(
            check_type=CheckType.REPUTATION,
            status="completed",
            risk_level=RiskLevel.LOW,
            score=80.0,
            findings=["Репутационных рисков не выявлено"],
            recommendations=["Мониторить отзывы и новости о компании"]
        )

    def _create_empty_check(self, check_type: CheckType) -> DueDiligenceCheck:
        """
        Создание пустой проверки.
        """
        return DueDiligenceCheck(
            check_type=check_type,
            status="pending",
            risk_level=RiskLevel.LOW,
            score=50.0,
            findings=["Проверка не выполнялась"],
            recommendations=["Выполнить проверку"]
        )

    def _create_failed_check(self, check_type: CheckType, error: str) -> DueDiligenceCheck:
        """
        Создание проверки с ошибкой.
        """
        return DueDiligenceCheck(
            check_type=check_type,
            status="failed",
            risk_level=RiskLevel.HIGH,
            score=0.0,
            findings=[f"Ошибка проверки: {error}"],
            recommendations=["Повторить проверку или выполнить вручную"],
            error_message=error
        )

    def _generate_overall_recommendations(self, report: DueDiligenceReport) -> List[str]:
        """
        Генерация общих рекомендаций на основе всего отчета.
        """
        recommendations = []

        # Рекомендации по уровню риска
        if report.overall_risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Немедленно отказаться от сделки с контрагентом",
                "Провести дополнительное расследование",
                "Обратиться к специалистам по due diligence"
            ])
        elif report.overall_risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Тщательно взвесить риски перед заключением сделки",
                "Включить дополнительные гарантии в договор",
                "Установить лимиты на сумму сделки"
            ])
        elif report.overall_risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Провести дополнительную проверку по ключевым рискам",
                "Включить стандартные гарантии в договор",
                "Установить мониторинг на период сделки"
            ])
        else:
            recommendations.append("Можно заключать сделку с стандартными условиями")

        # Рекомендации по конкретным проверкам
        if report.sanctions.risk_level == RiskLevel.CRITICAL:
            recommendations.append("Приоритет: проверить санкционные риски")

        if report.court_cases.score < 50:
            recommendations.append("Обратить внимание на судебную историю")

        return recommendations

    def _identify_risk_factors(self, report: DueDiligenceReport) -> List[str]:
        """
        Выявление факторов риска.
        """
        risk_factors = []

        # Анализируем каждую проверку
        if report.sanctions.risk_level == RiskLevel.CRITICAL:
            risk_factors.append("Нахождение в санкционных списках")

        if report.court_cases.score < 60:
            risk_factors.append("Неблагоприятная судебная история")

        if report.tax_debt.risk_level == RiskLevel.HIGH:
            risk_factors.append("Налоговые задолженности")

        if report.affiliations.risk_level == RiskLevel.HIGH:
            risk_factors.append("Риски связанные с аффилированными лицами")

        if report.licenses.risk_level == RiskLevel.HIGH:
            risk_factors.append("Отсутствие необходимых лицензий")

        if report.reputation.risk_level == RiskLevel.HIGH:
            risk_factors.append("Репутационные риски")

        if not risk_factors:
            risk_factors.append("Значительных рисков не выявлено")

        return risk_factors

    def _generate_deal_warnings(self, report: DueDiligenceReport) -> List[str]:
        """
        Генерация предупреждений для сделок.
        """
        warnings = []

        if report.overall_risk_level == RiskLevel.CRITICAL:
            warnings.append("Критический уровень риска - сделка не рекомендуется")

        if report.sanctions.risk_level == RiskLevel.CRITICAL:
            warnings.append("Санкционные ограничения - сделка невозможна")

        if report.court_cases.score < 40:
            warnings.append("Высокий судебный риск - предусмотреть гарантии")

        if report.company.status != "active":
            warnings.append(f"Статус компании: {report.company.status}")

        return warnings if warnings else ["Особых предупреждений нет"]
