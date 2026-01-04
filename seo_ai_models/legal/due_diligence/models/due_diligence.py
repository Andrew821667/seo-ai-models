"""
Модели данных для системы due diligence контрагентов.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Уровни риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DueDiligenceStatus(str, Enum):
    """Статусы проверки due diligence."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckType(str, Enum):
    """Типы проверок."""
    COURT_CASES = "court_cases"      # Судебные дела
    SANCTIONS = "sanctions"          # Санкционные списки
    TAX_DEBT = "tax_debt"           # Налоговые задолженности
    LIQUIDATION = "liquidation"     # Ликвидация/банкротство
    AFFILIATES = "affiliates"       # Аффилированные лица
    LICENSES = "licenses"           # Лицензии и разрешения
    REGISTRATION = "registration"   # Государственная регистрация


@dataclass
class CompanyInfo:
    """Информация о компании."""

    name: str
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    kpp: Optional[str] = None
    address: Optional[str] = None
    director: Optional[str] = None
    registration_date: Optional[datetime] = None
    status: str = "active"  # active, liquidated, bankrupt
    capital: Optional[float] = None
    main_activity: Optional[str] = None


@dataclass
class CourtCaseSummary:
    """Сводка по судебным делам контрагента."""

    total_cases: int = 0
    plaintiff_cases: int = 0
    defendant_cases: int = 0
    won_cases: int = 0
    lost_cases: int = 0
    pending_cases: int = 0
    total_amount: float = 0.0
    average_amount: float = 0.0
    recent_cases: List[Dict[str, Any]] = field(default_factory=list)  # последние 5 дел


@dataclass
class SanctionsCheck:
    """Результат проверки санкционных списков."""

    is_sanctioned: bool = False
    sanction_type: Optional[str] = None
    sanction_country: Optional[str] = None
    sanction_date: Optional[datetime] = None
    sanction_reason: Optional[str] = None
    checked_lists: List[str] = field(default_factory=list)  # проверенные списки


@dataclass
class TaxDebtCheck:
    """Результат проверки налоговых задолженностей."""

    has_debt: bool = False
    total_debt: float = 0.0
    debt_details: List[Dict[str, Any]] = field(default_factory=list)
    last_check_date: Optional[datetime] = None


@dataclass
class AffiliationCheck:
    """Результат проверки аффилированных лиц."""

    affiliates_found: int = 0
    key_affiliates: List[Dict[str, Any]] = field(default_factory=list)
    ownership_structure: Dict[str, Any] = field(default_factory=dict)
    beneficial_owners: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LicenseCheck:
    """Результат проверки лицензий и разрешений."""

    has_required_licenses: bool = True
    missing_licenses: List[str] = field(default_factory=list)
    expired_licenses: List[str] = field(default_factory=list)
    license_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReputationAnalysis:
    """Анализ репутации контрагента."""

    overall_score: float = 0.0  # 0-100
    reputation_factors: Dict[str, float] = field(default_factory=dict)
    negative_signals: List[str] = field(default_factory=list)
    positive_signals: List[str] = field(default_factory=list)
    news_mentions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DueDiligenceCheck:
    """Результат отдельной проверки."""

    check_type: CheckType
    status: str = "pending"  # pending, in_progress, completed, failed
    risk_level: RiskLevel = RiskLevel.LOW
    score: float = 0.0  # 0-100, где 100 - идеально
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    checked_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class DueDiligenceReport:
    """Полный отчет due diligence."""

    company: CompanyInfo
    overall_risk_level: RiskLevel
    overall_score: float  # 0-100
    summary: str

    # Результаты проверок
    court_cases: DueDiligenceCheck
    sanctions: DueDiligenceCheck
    tax_debt: DueDiligenceCheck
    affiliations: DueDiligenceCheck
    licenses: DueDiligenceCheck
    reputation: DueDiligenceCheck

    # Общие рекомендации
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    deal_warnings: List[str] = field(default_factory=list)

    # Метаданные
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checked_by: Optional[str] = None

    def __post_init__(self):
        """Обновление даты изменения."""
        self.updated_at = datetime.now()

    def calculate_overall_score(self):
        """Расчет общего скора на основе всех проверок."""
        checks = [self.court_cases, self.sanctions, self.tax_debt,
                 self.affiliations, self.licenses, self.reputation]

        # Веса для разных типов проверок
        weights = {
            CheckType.COURT_CASES: 0.25,
            CheckType.SANCTIONS: 0.30,  # высокий вес для санкций
            CheckType.TAX_DEBT: 0.15,
            CheckType.AFFILIATES: 0.10,
            CheckType.LICENSES: 0.10,
            CheckType.REPUTATION: 0.10
        }

        total_score = 0.0
        total_weight = 0.0

        for check in checks:
            if check.status == "completed":
                weight = weights.get(check.check_type, 0.1)
                total_score += check.score * weight
                total_weight += weight

        self.overall_score = total_score / total_weight if total_weight > 0 else 50.0

        # Определение общего уровня риска
        if self.overall_score >= 80:
            self.overall_risk_level = RiskLevel.LOW
        elif self.overall_score >= 60:
            self.overall_risk_level = RiskLevel.MEDIUM
        elif self.overall_score >= 40:
            self.overall_risk_level = RiskLevel.HIGH
        else:
            self.overall_risk_level = RiskLevel.CRITICAL

    def generate_summary(self):
        """Генерация текстового резюме отчета."""
        risk_descriptions = {
            RiskLevel.LOW: "Низкий уровень риска",
            RiskLevel.MEDIUM: "Средний уровень риска",
            RiskLevel.HIGH: "Высокий уровень риска",
            RiskLevel.CRITICAL: "Критический уровень риска"
        }

        self.summary = f"""
Отчет due diligence для {self.company.name}

Общий уровень риска: {risk_descriptions[self.overall_risk_level]}
Общий скор: {self.overall_score:.1f}/100

Ключевые результаты:
- Судебные дела: {self.court_cases.findings[:2] if self.court_cases.findings else ['Без замечаний']}
- Санкции: {'Найдены' if self.sanctions.data.get('is_sanctioned') else 'Не найдены'}
- Налоговые задолженности: {'Есть' if self.tax_debt.data.get('has_debt') else 'Нет'}
- Аффилированные лица: {len(self.affiliations.data.get('beneficial_owners', []))} бенефициаров

Рекомендации: {len(self.recommendations)} пунктов
        """.strip()


@dataclass
class DueDiligenceRequest:
    """Запрос на проведение due diligence."""

    company_name: str
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    priority: str = "normal"  # normal, urgent, express
    checks_to_perform: List[CheckType] = field(default_factory=lambda: [
        CheckType.COURT_CASES, CheckType.SANCTIONS, CheckType.TAX_DEBT
    ])
    additional_context: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    requested_by: Optional[str] = None
