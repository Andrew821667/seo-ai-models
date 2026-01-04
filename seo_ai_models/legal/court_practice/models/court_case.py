"""
Модели данных для судебных дел в системе анализа судебной практики.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class CourtType(str, Enum):
    """Типы судов."""
    ARBITRAGE = "arbitrage"  # Арбитражный суд
    GENERAL = "general"     # Суд общей юрисдикции
    SUPREME = "supreme"     # Верховный суд


class CaseStatus(str, Enum):
    """Статусы судебных дел."""
    PENDING = "pending"         # В производстве
    DECIDED = "decided"          # Решено
    APPEALED = "appealed"        # Обжаловано
    CLOSED = "closed"           # Закрыто
    SUSPENDED = "suspended"     # Приостановлено


class CaseCategory(str, Enum):
    """Категории судебных дел."""
    CONTRACT = "contract"           # Договорные споры
    PROPERTY = "property"           # Имущественные споры
    CORPORATE = "corporate"         # Корпоративные споры
    TAX = "tax"                     # Налоговые споры
    LABOR = "labor"                 # Трудовые споры
    ADMINISTRATIVE = "administrative" # Административные споры
    OTHER = "other"                 # Прочие


@dataclass
class Party:
    """Участник судебного дела."""

    name: str
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    address: Optional[str] = None
    role: str = "plaintiff"  # plaintiff, defendant, third_party
    representative: Optional[str] = None


@dataclass
class CourtDecision:
    """Судебное решение."""

    date: datetime
    court: str
    judge: Optional[str] = None
    decision_type: str = "decision"  # decision, ruling, definition
    outcome: Optional[str] = None  # satisfied, dismissed, partially_satisfied
    amount: Optional[float] = None
    text_summary: Optional[str] = None
    full_text: Optional[str] = None


@dataclass
class CourtCase:
    """Судебное дело."""

    case_number: str
    court_type: CourtType
    court_name: str
    category: CaseCategory
    status: CaseStatus

    # Участники
    plaintiffs: List[Party] = field(default_factory=list)
    defendants: List[Party] = field(default_factory=list)
    third_parties: List[Party] = field(default_factory=list)

    # Суть спора
    claim_subject: str = ""
    claim_amount: Optional[float] = None
    description: str = ""

    # Хронология
    filing_date: Optional[datetime] = None
    hearing_dates: List[datetime] = field(default_factory=list)
    decisions: List[CourtDecision] = field(default_factory=list)

    # Анализ
    keywords: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high
    similar_cases: List[str] = field(default_factory=list)  # номера похожих дел

    # Метаданные
    source_url: Optional[str] = None
    parsed_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Валидация данных после инициализации."""
        if not self.case_number:
            raise ValueError("Номер дела обязателен")

        # Обновляем дату последнего изменения
        self.last_updated = datetime.now()


@dataclass
class CaseAnalysis:
    """Результат анализа судебного дела."""

    case: CourtCase
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    similar_cases_analysis: List[Dict[str, Any]] = field(default_factory=list)
    predictive_outcome: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class CourtPracticeReport:
    """Отчет по судебной практике."""

    query: str
    cases_found: int
    cases_analyzed: int
    analysis_results: List[CaseAnalysis] = field(default_factory=list)

    # Статистика
    category_distribution: Dict[str, int] = field(default_factory=dict)
    outcome_statistics: Dict[str, int] = field(default_factory=dict)
    average_decision_time: Optional[float] = None  # в днях

    # Тренды
    trends: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    generated_at: datetime = field(default_factory=datetime.now)
