"""
Модели данных для системы автоматизации претензионно-исковой работы.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum


class ClaimType(str, Enum):
    """Типы претензий."""
    CONTRACT_BREACH = "contract_breach"      # Нарушение условий договора
    PAYMENT_DELAY = "payment_delay"          # Просрочка платежа
    QUALITY_DEFECT = "quality_defect"        # Дефекты качества
    DELIVERY_DELAY = "delivery_delay"        # Просрочка поставки
    PROPERTY_DAMAGE = "property_damage"      # Повреждение имущества
    SERVICE_DEFECT = "service_defect"        # Недостатки услуг
    OTHER = "other"                         # Прочие


class ClaimStatus(str, Enum):
    """Статусы претензионно-исковой работы."""
    DRAFT = "draft"                         # Черновик
    SENT = "sent"                           # Отправлена
    RECEIVED = "received"                   # Получена ответчиком
    RESPONDED = "responded"                 # Получен ответ
    SATISFIED = "satisfied"                 # Удовлетворена
    PARTIALLY_SATISFIED = "partially_satisfied"  # Частично удовлетворена
    REJECTED = "rejected"                   # Отклонена
    ESCALATED_TO_COURT = "escalated_to_court"  # Передана в суд
    CLOSED = "closed"                       # Закрыта


class CourtClaimType(str, Enum):
    """Типы исковых заявлений."""
    ECONOMIC_DISPUTE = "economic_dispute"   # Экономический спор
    CONTRACT_CLAIM = "contract_claim"       # Иск по договору
    PROPERTY_CLAIM = "property_claim"       # Виндикационный иск
    TORT_CLAIM = "tort_claim"              # Деликтный иск
    ADMINISTRATIVE_CLAIM = "administrative_claim"  # Административный иск


class LimitationPeriodType(str, Enum):
    """Типы сроков исковой давности."""
    GENERAL = "general"                     # Общий срок (3 года)
    CONTRACT = "contract"                   # По договорам (3 года)
    PROPERTY = "property"                   # Виндикационный (3 года)
    TORT = "tort"                          # Деликтный (3 года)
    ADMINISTRATIVE = "administrative"       # Административный (3 месяца)
    SHORT = "short"                        # Краткий срок (1 год)
    SPECIAL = "special"                     # Специальный срок


@dataclass
class Party:
    """Участник спора."""
    name: str
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    address: str = ""
    representative: Optional[str] = None
    contact_info: Optional[str] = None
    role: str = "claimant"  # claimant, respondent, third_party


@dataclass
class ClaimEvent:
    """Событие в претензионно-исковой работе."""
    event_type: str  # sent, received, responded, escalated
    description: str
    date: datetime
    documents: List[str] = field(default_factory=list)  # имена файлов
    notes: Optional[str] = None


@dataclass
class Claim:
    """Претензия."""
    claim_id: str
    type: ClaimType
    title: str
    claimant: Party
    respondent: Party
    description: str
    amount: Optional[float] = None
    currency: str = "RUB"
    due_date: Optional[datetime] = None  # срок удовлетворения претензии

    # Статус и история
    status: ClaimStatus = ClaimStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Документы
    claim_text: Optional[str] = None
    attachments: List[str] = field(default_factory=list)

    # История событий
    events: List[ClaimEvent] = field(default_factory=list)

    # Анализ
    success_probability: Optional[float] = None
    risk_level: str = "medium"  # low, medium, high
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Валидация данных после инициализации."""
        if not self.claim_id:
            raise ValueError("ID претензии обязателен")

        if not self.title:
            raise ValueError("Заголовок претензии обязателен")


@dataclass
class CourtClaim:
    """Исковое заявление."""
    claim_id: str
    type: CourtClaimType
    title: str
    claimant: Party
    respondent: Party
    third_parties: List[Party] = field(default_factory=list)

    # Предмет иска
    claim_subject: str
    claim_amount: Optional[float] = None
    currency: str = "RUB"

    # Обстоятельства дела
    circumstances: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Ссылки на нормы права
    legal_references: List[str] = field(default_factory=list)

    # Расчет исковых требований
    calculation: Optional[str] = None

    # Статус
    status: ClaimStatus = ClaimStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    filed_at: Optional[datetime] = None
    court_case_number: Optional[str] = None

    # Документы
    claim_text: Optional[str] = None
    attachments: List[str] = field(default_factory=list)

    # Сроки
    limitation_period_type: LimitationPeriodType = LimitationPeriodType.GENERAL
    limitation_deadline: Optional[datetime] = None

    # Анализ перспектив
    success_probability: Optional[float] = None
    estimated_duration: Optional[int] = None  # в днях
    estimated_cost: Optional[float] = None

    def calculate_limitation_deadline(self, start_date: datetime) -> datetime:
        """
        Расчет срока исковой давности.
        """
        if self.limitation_period_type == LimitationPeriodType.GENERAL:
            return start_date + timedelta(days=1095)  # 3 года
        elif self.limitation_period_type == LimitationPeriodType.CONTRACT:
            return start_date + timedelta(days=1095)  # 3 года
        elif self.limitation_period_type == LimitationPeriodType.PROPERTY:
            return start_date + timedelta(days=1095)  # 3 года
        elif self.limitation_period_type == LimitationPeriodType.TORT:
            return start_date + timedelta(days=1095)  # 3 года
        elif self.limitation_period_type == LimitationPeriodType.ADMINISTRATIVE:
            return start_date + timedelta(days=90)    # 3 месяца
        elif self.limitation_period_type == LimitationPeriodType.SHORT:
            return start_date + timedelta(days=365)   # 1 год
        else:
            return start_date + timedelta(days=1095)  # по умолчанию 3 года


@dataclass
class ClaimAnalysis:
    """Результат анализа претензии/иска."""
    claim: Union[Claim, CourtClaim]
    legal_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    success_prediction: Dict[str, Any] = field(default_factory=dict)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class LimitationPeriod:
    """Срок исковой давности."""
    period_type: LimitationPeriodType
    start_date: datetime
    duration_days: int
    deadline: datetime
    remaining_days: int

    # Причины прерывания/приостановления
    interruptions: List[Dict[str, Any]] = field(default_factory=list)
    suspensions: List[Dict[str, Any]] = field(default_factory=list)

    # Текущий статус
    is_expired: bool = False
    is_suspended: bool = False
    last_activity_date: Optional[datetime] = None

    def update_remaining_days(self):
        """Обновление количества оставшихся дней."""
        now = datetime.now()
        if self.is_suspended:
            # Если срок приостановлен, не меняем remaining_days
            return

        if now >= self.deadline:
            self.remaining_days = 0
            self.is_expired = True
        else:
            self.remaining_days = (self.deadline - now).days
            self.is_expired = False

    def interrupt_period(self, interruption_date: datetime, reason: str):
        """
        Прерывание срока исковой давности.
        """
        self.interruptions.append({
            'date': interruption_date,
            'reason': reason,
            'previous_deadline': self.deadline
        })

        # После перерыва начинается новый срок
        self.start_date = interruption_date
        self.deadline = self.start_date + timedelta(days=self.duration_days)
        self.last_activity_date = interruption_date
        self.update_remaining_days()

    def suspend_period(self, suspension_start: datetime, reason: str):
        """
        Приостановление срока исковой давности.
        """
        self.suspensions.append({
            'start_date': suspension_start,
            'reason': reason
        })
        self.is_suspended = True


@dataclass
class ClaimsWorkflow:
    """Рабочий процесс претензионно-исковой работы."""
    workflow_id: str
    title: str
    claimant: Party
    respondent: Party

    # Этапы процесса
    stages: List[Dict[str, Any]] = field(default_factory=list)

    # Текущий статус
    current_stage: str = "initial"
    status: str = "active"

    # Документы и события
    documents: List[str] = field(default_factory=list)
    events: List[ClaimEvent] = field(default_factory=list)

    # Сроки
    deadlines: Dict[str, datetime] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_stage(self, stage_name: str, description: str, due_date: Optional[datetime] = None):
        """Добавление этапа в процесс."""
        stage = {
            'name': stage_name,
            'description': description,
            'status': 'pending',
            'created_at': datetime.now(),
            'due_date': due_date
        }
        self.stages.append(stage)
        if due_date:
            self.deadlines[stage_name] = due_date

    def complete_stage(self, stage_name: str):
        """Отметка этапа как выполненного."""
        for stage in self.stages:
            if stage['name'] == stage_name:
                stage['status'] = 'completed'
                stage['completed_at'] = datetime.now()
                break
        self.updated_at = datetime.now()

    def get_next_deadline(self) -> Optional[datetime]:
        """Получение ближайшего дедлайна."""
        future_deadlines = [
            deadline for deadline in self.deadlines.values()
            if deadline > datetime.now()
        ]
        return min(future_deadlines) if future_deadlines else None
