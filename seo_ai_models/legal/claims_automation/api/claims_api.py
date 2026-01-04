"""
API маршруты для системы автоматизации претензионно-исковой работы.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from seo_ai_models.api.auth.dependencies import get_current_user, require_permission
from seo_ai_models.api.auth.models import User, Permission
from seo_ai_models.api.websocket.manager import progress_tracker

from seo_ai_models.legal.claims_automation.models.claims_models import (
    Claim, CourtClaim, ClaimType, CourtClaimType, LimitationPeriodType, Party
)
from seo_ai_models.legal.claims_automation.generators.claims_generator import ClaimsGenerator
from seo_ai_models.legal.claims_automation.analyzers.claims_analyzer import ClaimsAnalyzer

# Создаем роутер
router = APIRouter(prefix="/claims-automation", tags=["claims-automation"])

# Инициализируем компоненты
claims_generator = ClaimsGenerator()
claims_analyzer = ClaimsAnalyzer()


# Модели запросов/ответов
class PartyModel(BaseModel):
    """Модель участника спора."""
    name: str = Field(..., description="Название организации/ФИО")
    inn: Optional[str] = Field(None, description="ИНН")
    ogrn: Optional[str] = Field(None, description="ОГРН")
    address: str = Field(..., description="Адрес")
    representative: Optional[str] = Field(None, description="Представитель")
    contact_info: Optional[str] = Field(None, description="Контактная информация")


class ClaimRequest(BaseModel):
    """Запрос на создание претензии."""
    claim_id: str = Field(..., description="Уникальный ID претензии")
    type: str = Field(..., description="Тип претензии")
    title: str = Field(..., description="Заголовок претензии")
    claimant: PartyModel
    respondent: PartyModel
    description: str = Field(..., description="Описание претензии")
    amount: Optional[float] = Field(None, description="Сумма претензии")
    currency: str = Field("RUB", description="Валюта")
    due_date: Optional[datetime] = Field(None, description="Срок удовлетворения")


class CourtClaimRequest(BaseModel):
    """Запрос на создание искового заявления."""
    claim_id: str = Field(..., description="Уникальный ID иска")
    type: str = Field(..., description="Тип искового заявления")
    title: str = Field(..., description="Заголовок иска")
    claimant: PartyModel
    respondent: PartyModel
    third_parties: List[PartyModel] = Field(default_factory=list, description="Третьи лица")
    claim_subject: str = Field(..., description="Предмет иска")
    claim_amount: Optional[float] = Field(None, description="Сумма иска")
    currency: str = Field("RUB", description="Валюта")
    circumstances: str = Field(..., description="Обстоятельства дела")
    evidence: List[dict] = Field(default_factory=list, description="Доказательства")
    legal_references: List[str] = Field(default_factory=list, description="Ссылки на нормы права")
    calculation: Optional[str] = Field(None, description="Расчет требований")
    limitation_period_type: str = Field("general", description="Тип срока исковой давности")


class ClaimAnalysisRequest(BaseModel):
    """Запрос на анализ претензии."""
    claim: ClaimRequest


class CourtClaimAnalysisRequest(BaseModel):
    """Запрос на анализ искового заявления."""
    court_claim: CourtClaimRequest


class GeneratedDocumentResponse(BaseModel):
    """Ответ с сгенерированным документом."""
    document_id: str
    document_type: str  # claim, court_claim, response
    content: str
    generated_at: datetime


class ClaimAnalysisResponse(BaseModel):
    """Ответ с анализом претензии."""
    analysis_id: str
    legal_analysis: dict
    risk_assessment: dict
    success_prediction: dict
    recommendations: List[str]
    similar_cases: List[dict]
    analyzed_at: datetime


@router.post("/generate-claim", response_model=GeneratedDocumentResponse)
async def generate_claim_document(
    request: ClaimRequest,
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Генерация текста претензии.
    """
    try:
        # Преобразуем тип претензии
        try:
            claim_type = ClaimType(request.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип претензии: {request.type}")

        # Создаем объект претензии
        claimant = Party(**request.claimant.dict())
        respondent = Party(**request.respondent.dict())

        claim = Claim(
            claim_id=request.claim_id,
            type=claim_type,
            title=request.title,
            claimant=claimant,
            respondent=respondent,
            description=request.description,
            amount=request.amount,
            currency=request.currency,
            due_date=request.due_date
        )

        # Генерируем текст
        document_content = claims_generator.generate_claim(claim)

        return GeneratedDocumentResponse(
            document_id=f"claim_{request.claim_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            document_type="claim",
            content=document_content,
            generated_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации претензии: {str(e)}")


@router.post("/generate-court-claim", response_model=GeneratedDocumentResponse)
async def generate_court_claim_document(
    request: CourtClaimRequest,
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Генерация текста искового заявления.
    """
    try:
        # Преобразуем типы
        try:
            claim_type = CourtClaimType(request.type)
            limitation_type = LimitationPeriodType(request.limitation_period_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип: {str(e)}")

        # Создаем объект искового заявления
        claimant = Party(**request.claimant.dict())
        respondent = Party(**request.respondent.dict())
        third_parties = [Party(**tp.dict()) for tp in request.third_parties]

        court_claim = CourtClaim(
            claim_id=request.claim_id,
            type=claim_type,
            title=request.title,
            claimant=claimant,
            respondent=respondent,
            third_parties=third_parties,
            claim_subject=request.claim_subject,
            claim_amount=request.claim_amount,
            currency=request.currency,
            circumstances=request.circumstances,
            evidence=request.evidence,
            legal_references=request.legal_references,
            calculation=request.calculation,
            limitation_period_type=limitation_type
        )

        # Генерируем текст
        document_content = claims_generator.generate_court_claim(court_claim)

        return GeneratedDocumentResponse(
            document_id=f"court_claim_{request.claim_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            document_type="court_claim",
            content=document_content,
            generated_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации искового заявления: {str(e)}")


@router.post("/analyze-claim", response_model=ClaimAnalysisResponse)
async def analyze_claim(
    request: ClaimAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    analysis_id: Optional[str] = None
):
    """
    Анализ претензии.
    """
    try:
        # Преобразуем тип претензии
        try:
            claim_type = ClaimType(request.claim.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип претензии: {request.claim.type}")

        # Создаем объект претензии
        claimant = Party(**request.claim.claimant.dict())
        respondent = Party(**request.claim.respondent.dict())

        claim = Claim(
            claim_id=request.claim.claim_id,
            type=claim_type,
            title=request.claim.title,
            claimant=claimant,
            respondent=respondent,
            description=request.claim.description,
            amount=request.claim.amount,
            currency=request.claim.currency,
            due_date=request.claim.due_date
        )

        # Генерируем ID анализа
        if not analysis_id:
            analysis_id = f"claim_analysis_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(analysis_id, f"Анализ претензии {claim.claim_id}")

        # Выполняем анализ в фоне
        background_tasks.add_task(
            _perform_claim_analysis,
            analysis_id,
            claim
        )

        # Возвращаем начальный статус
        return await get_claim_analysis_status(analysis_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа претензии: {str(e)}")


@router.post("/analyze-court-claim", response_model=ClaimAnalysisResponse)
async def analyze_court_claim(
    request: CourtClaimAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    analysis_id: Optional[str] = None
):
    """
    Анализ искового заявления.
    """
    try:
        # Преобразуем типы
        try:
            claim_type = CourtClaimType(request.court_claim.type)
            limitation_type = LimitationPeriodType(request.court_claim.limitation_period_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип: {str(e)}")

        # Создаем объект искового заявления
        claimant = Party(**request.court_claim.claimant.dict())
        respondent = Party(**request.court_claim.respondent.dict())
        third_parties = [Party(**tp.dict()) for tp in request.court_claim.third_parties]

        court_claim = CourtClaim(
            claim_id=request.court_claim.claim_id,
            type=claim_type,
            title=request.court_claim.title,
            claimant=claimant,
            respondent=respondent,
            third_parties=third_parties,
            claim_subject=request.court_claim.claim_subject,
            claim_amount=request.court_claim.claim_amount,
            currency=request.court_claim.currency,
            circumstances=request.court_claim.circumstances,
            evidence=request.court_claim.evidence,
            legal_references=request.court_claim.legal_references,
            calculation=request.court_claim.calculation,
            limitation_period_type=limitation_type
        )

        # Генерируем ID анализа
        if not analysis_id:
            analysis_id = f"court_claim_analysis_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(analysis_id, f"Анализ искового заявления {court_claim.claim_id}")

        # Выполняем анализ в фоне
        background_tasks.add_task(
            _perform_court_claim_analysis,
            analysis_id,
            court_claim
        )

        # Возвращаем начальный статус
        return await get_claim_analysis_status(analysis_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа искового заявления: {str(e)}")


@router.get("/analysis/status/{analysis_id}")
async def get_claim_analysis_status(analysis_id: str):
    """
    Получение статуса анализа претензии/иска.
    """
    try:
        status = progress_tracker.get_status(analysis_id)
        if not status:
            raise HTTPException(status_code=404, detail="Анализ не найден")

        if status.get("status") == "completed":
            # Возвращаем результаты анализа
            analysis_result = status.get("results")
            if not analysis_result:
                raise HTTPException(status_code=500, detail="Результаты анализа не найдены")

            return ClaimAnalysisResponse(
                analysis_id=analysis_id,
                legal_analysis=analysis_result.legal_analysis,
                risk_assessment=analysis_result.risk_assessment,
                success_prediction=analysis_result.success_prediction,
                recommendations=analysis_result.recommendations,
                similar_cases=analysis_result.similar_cases,
                analyzed_at=analysis_result.analyzed_at
            )
        else:
            # Возвращаем текущий статус
            return {
                "status": status.get("status", "unknown"),
                "progress": status.get("progress", 0),
                "current_step": status.get("current_step", ""),
                "message": status.get("message", "")
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса анализа: {str(e)}")


@router.post("/generate-response")
async def generate_claim_response(
    claim_id: str,
    response_type: str = "rejection",  # satisfaction, partial_satisfaction, rejection
    response_text: str = "",
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Генерация ответа на претензию.
    """
    try:
        # В реальной реализации здесь нужно получить оригинальную претензию из базы данных
        # Пока создаем примерную претензию для демонстрации

        # Пример: создаем базовую претензию (в продакшене загрузить из БД)
        original_claim = Claim(
            claim_id=claim_id,
            type=ClaimType.CONTRACT_BREACH,
            title="Претензия о нарушении условий договора",
            claimant=Party(name="Истец", address="Адрес истца"),
            respondent=Party(name="Ответчик", address="Адрес ответчика"),
            description="Описание претензии"
        )

        # Генерируем ответ
        response_content = claims_generator.generate_response_to_claim(
            original_claim, response_type, response_text
        )

        return GeneratedDocumentResponse(
            document_id=f"response_{claim_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            document_type="response",
            content=response_content,
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")


@router.get("/limitation-period/check")
async def check_limitation_period(
    start_date: datetime,
    period_type: str = "general",
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Проверка срока исковой давности.
    """
    try:
        # Преобразуем тип периода
        try:
            limitation_type = LimitationPeriodType(period_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип срока: {period_type}")

        # Создаем объект периода
        temp_claim = CourtClaim(
            claim_id="temp",
            type=CourtClaimType.ECONOMIC_DISPUTE,
            title="temp",
            claimant=Party(name="temp"),
            respondent=Party(name="temp"),
            claim_subject="temp",
            circumstances="temp",
            limitation_period_type=limitation_type
        )

        # Анализируем срок
        limitation = claims_analyzer.analyze_limitation_period(temp_claim)

        return {
            'period_type': limitation.period_type.value,
            'start_date': limitation.start_date.isoformat(),
            'duration_days': limitation.duration_days,
            'deadline': limitation.deadline.isoformat(),
            'remaining_days': limitation.remaining_days,
            'is_expired': limitation.is_expired,
            'is_suspended': limitation.is_suspended
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка проверки срока давности: {str(e)}")


# Вспомогательные функции для фоновой обработки

async def _perform_claim_analysis(analysis_id: str, claim: Claim):
    """
    Выполнение анализа претензии в фоне.
    """
    try:
        progress_tracker.update_progress(analysis_id, 20, "Юридический анализ")

        # Выполняем анализ
        analysis_result = claims_analyzer.analyze_claim(claim)

        progress_tracker.update_progress(analysis_id, 80, "Формирование рекомендаций")

        # Сохраняем результаты
        progress_tracker.complete_tracking(analysis_id, results=analysis_result)

    except Exception as e:
        progress_tracker.fail_tracking(analysis_id, f"Ошибка анализа претензии: {str(e)}")


async def _perform_court_claim_analysis(analysis_id: str, court_claim: CourtClaim):
    """
    Выполнение анализа искового заявления в фоне.
    """
    try:
        progress_tracker.update_progress(analysis_id, 20, "Анализ подсудности")

        # Выполняем анализ
        analysis_result = claims_analyzer.analyze_court_claim(court_claim)

        progress_tracker.update_progress(analysis_id, 80, "Анализ сроков давности")

        # Сохраняем результаты
        progress_tracker.complete_tracking(analysis_id, results=analysis_result)

    except Exception as e:
        progress_tracker.fail_tracking(analysis_id, f"Ошибка анализа искового заявления: {str(e)}")
