"""
API маршруты для системы due diligence контрагентов.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from seo_ai_models.api.auth.dependencies import get_current_user, require_permission
from seo_ai_models.api.auth.models import User, Permission
from seo_ai_models.api.websocket.manager import progress_tracker

from seo_ai_models.legal.due_diligence.models.due_diligence import (
    DueDiligenceRequest, DueDiligenceReport, CheckType
)
from seo_ai_models.legal.due_diligence.analyzers.due_diligence_analyzer import DueDiligenceAnalyzer

# Создаем роутер
router = APIRouter(prefix="/due-diligence", tags=["due-diligence"])

# Инициализируем анализатор
due_diligence_analyzer = DueDiligenceAnalyzer()


# Модели запросов/ответов
class DueDiligenceRequestModel(BaseModel):
    """Запрос на проведение due diligence."""

    company_name: str = Field(..., description="Название компании для проверки")
    inn: Optional[str] = Field(None, description="ИНН компании")
    ogrn: Optional[str] = Field(None, description="ОГРН компании")
    priority: str = Field("normal", description="Приоритет проверки (normal, urgent, express)")
    checks_to_perform: List[str] = Field(
        default=["court_cases", "sanctions"],
        description="Типы проверок для выполнения"
    )
    additional_context: Optional[str] = Field(None, description="Дополнительная информация")


class DueDiligenceResponse(BaseModel):
    """Ответ с результатами due diligence."""

    report_id: str
    company_name: str
    overall_risk_level: str
    overall_score: float
    summary: str
    recommendations: List[str]
    risk_factors: List[str]
    deal_warnings: List[str]
    created_at: datetime
    checked_by: Optional[str]


class DueDiligenceStatusResponse(BaseModel):
    """Ответ со статусом проверки due diligence."""

    report_id: str
    status: str
    progress: int
    current_step: str
    estimated_completion: Optional[datetime]


class DueDiligenceDetailedReport(BaseModel):
    """Детальный отчет due diligence."""

    company: dict
    overall_risk_level: str
    overall_score: float
    summary: str

    # Результаты проверок
    court_cases: dict
    sanctions: dict
    tax_debt: dict
    affiliations: dict
    licenses: dict
    reputation: dict

    recommendations: List[str]
    risk_factors: List[str]
    deal_warnings: List[str]

    created_at: datetime
    updated_at: datetime
    checked_by: Optional[str]


@router.post("/", response_model=DueDiligenceResponse)
async def create_due_diligence_check(
    request: DueDiligenceRequestModel,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    report_id: Optional[str] = None
):
    """
    Создание новой проверки due diligence контрагента.
    """
    try:
        # Преобразуем типы проверок
        checks_to_perform = []
        for check_str in request.checks_to_perform:
            try:
                checks_to_perform.append(CheckType(check_str))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неизвестный тип проверки: {check_str}"
                )

        # Создаем запрос
        dd_request = DueDiligenceRequest(
            company_name=request.company_name,
            inn=request.inn,
            ogrn=request.ogrn,
            priority=request.priority,
            checks_to_perform=checks_to_perform,
            additional_context=request.additional_context,
            requested_by=current_user.username
        )

        # Генерируем ID отчета если не указан
        if not report_id:
            report_id = f"dd_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(
            report_id,
            f"Due diligence для {request.company_name}"
        )

        # Выполняем проверку в фоне
        background_tasks.add_task(
            _perform_due_diligence_check,
            report_id,
            dd_request,
            current_user.username
        )

        # Возвращаем начальный статус
        return await get_due_diligence_status(report_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка создания проверки: {str(e)}")


@router.get("/status/{report_id}")
async def get_due_diligence_status(report_id: str):
    """
    Получение статуса проверки due diligence.
    """
    try:
        status = progress_tracker.get_status(report_id)
        if not status:
            raise HTTPException(status_code=404, detail="Проверка не найдена")

        if status.get("status") == "completed":
            # Возвращаем результаты
            report = status.get("results")
            if not report:
                raise HTTPException(status_code=500, detail="Результаты проверки не найдены")

            return DueDiligenceResponse(
                report_id=report_id,
                company_name=report.company.name,
                overall_risk_level=report.overall_risk_level.value,
                overall_score=report.overall_score,
                summary=report.summary,
                recommendations=report.recommendations,
                risk_factors=report.risk_factors,
                deal_warnings=report.deal_warnings,
                created_at=report.created_at,
                checked_by=report.checked_by
            )
        else:
            # Возвращаем текущий статус
            return DueDiligenceStatusResponse(
                report_id=report_id,
                status=status.get("status", "unknown"),
                progress=status.get("progress", 0),
                current_step=status.get("current_step", ""),
                estimated_completion=None  # можно добавить расчет
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.get("/report/{report_id}", response_model=DueDiligenceDetailedReport)
async def get_due_diligence_report(
    report_id: str,
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Получение детального отчета due diligence.
    """
    try:
        status = progress_tracker.get_status(report_id)
        if not status:
            raise HTTPException(status_code=404, detail="Отчет не найден")

        if status.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Проверка еще не завершена. Статус: {status.get('status')}"
            )

        # Получаем полный отчет
        report = status.get("results")
        if not report:
            raise HTTPException(status_code=500, detail="Данные отчета не найдены")

        return DueDiligenceDetailedReport(
            company={
                'name': report.company.name,
                'inn': report.company.inn,
                'ogrn': report.company.ogrn,
                'director': report.company.director,
                'address': report.company.address,
                'status': report.company.status
            },
            overall_risk_level=report.overall_risk_level.value,
            overall_score=report.overall_score,
            summary=report.summary,
            court_cases={
                'status': report.court_cases.status,
                'risk_level': report.court_cases.risk_level.value,
                'score': report.court_cases.score,
                'findings': report.court_cases.findings,
                'recommendations': report.court_cases.recommendations
            },
            sanctions={
                'status': report.sanctions.status,
                'risk_level': report.sanctions.risk_level.value,
                'score': report.sanctions.score,
                'findings': report.sanctions.findings,
                'recommendations': report.sanctions.recommendations
            },
            tax_debt={
                'status': report.tax_debt.status,
                'risk_level': report.tax_debt.risk_level.value,
                'score': report.tax_debt.score,
                'findings': report.tax_debt.findings,
                'recommendations': report.tax_debt.recommendations
            },
            affiliations={
                'status': report.affiliations.status,
                'risk_level': report.affiliations.risk_level.value,
                'score': report.affiliations.score,
                'findings': report.affiliations.findings,
                'recommendations': report.affiliations.recommendations
            },
            licenses={
                'status': report.licenses.status,
                'risk_level': report.licenses.risk_level.value,
                'score': report.licenses.score,
                'findings': report.licenses.findings,
                'recommendations': report.licenses.recommendations
            },
            reputation={
                'status': report.reputation.status,
                'risk_level': report.reputation.risk_level.value,
                'score': report.reputation.score,
                'findings': report.reputation.findings,
                'recommendations': report.reputation.recommendations
            },
            recommendations=report.recommendations,
            risk_factors=report.risk_factors,
            deal_warnings=report.deal_warnings,
            created_at=report.created_at,
            updated_at=report.updated_at,
            checked_by=report.checked_by
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения отчета: {str(e)}")


@router.get("/companies/{inn}/quick-check")
async def quick_company_check(
    inn: str,
    current_user: User = Depends(require_permission(Permission.ANALYST))
):
    """
    Быстрая проверка компании по ИНН (судебные дела + санкции).
    """
    try:
        # Выполняем быструю проверку
        quick_request = DueDiligenceRequest(
            company_name="",  # будет заполнено автоматически
            inn=inn,
            checks_to_perform=[CheckType.COURT_CASES, CheckType.SANCTIONS]
        )

        # Получаем информацию о компании
        company_info = due_diligence_analyzer._get_company_info(quick_request)

        # Выполняем только ключевые проверки
        checks_results = due_diligence_analyzer._perform_checks(quick_request, company_info)

        # Формируем краткий отчет
        court_check = checks_results.get(CheckType.COURT_CASES)
        sanctions_check = checks_results.get(CheckType.SANCTIONS)

        overall_risk = "low"
        overall_score = 90.0

        if sanctions_check and sanctions_check.risk_level.value == "critical":
            overall_risk = "critical"
            overall_score = 0.0
        elif court_check and court_check.score < 60:
            overall_risk = "medium"
            overall_score = court_check.score

        return {
            'inn': inn,
            'company_name': company_info.name,
            'overall_risk': overall_risk,
            'overall_score': overall_score,
            'court_cases': {
                'cases_found': court_check.data.get('cases_found', 0) if court_check else 0,
                'risk_level': court_check.risk_level.value if court_check else 'unknown'
            },
            'sanctions': {
                'is_sanctioned': sanctions_check.data.get('overall_result', {}).get('is_sanctioned', False) if sanctions_check else False,
                'risk_level': sanctions_check.risk_level.value if sanctions_check else 'unknown'
            },
            'checked_at': datetime.now()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка быстрой проверки: {str(e)}")


# Вспомогательные функции для фоновой обработки

async def _perform_due_diligence_check(
    report_id: str,
    request: DueDiligenceRequest,
    user_id: str
):
    """
    Выполнение проверки due diligence в фоне.
    """
    try:
        progress_tracker.update_progress(report_id, 10, "Подготовка проверки")

        # Выполняем проверку
        progress_tracker.update_progress(report_id, 20, "Сбор информации о компании")

        report = due_diligence_analyzer.perform_due_diligence(request, user_id)

        progress_tracker.update_progress(report_id, 90, "Формирование отчета")

        # Сохраняем результаты
        progress_tracker.complete_tracking(report_id, results=report)

    except Exception as e:
        progress_tracker.fail_tracking(report_id, f"Ошибка выполнения проверки: {str(e)}")
