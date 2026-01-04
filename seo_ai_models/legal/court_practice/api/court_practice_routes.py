"""
API маршруты для системы анализа судебной практики.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from datetime import datetime

from seo_ai_models.api.auth.dependencies import get_current_user, require_permission
from seo_ai_models.api.auth.models import User, Permission
from seo_ai_models.api.infrastructure.database import get_db
from seo_ai_models.api.websocket.manager import progress_tracker

from seo_ai_models.legal.court_practice.parsers.arbitrage_court_parser import ArbitrageCourtParser
from seo_ai_models.legal.court_practice.analyzers.court_case_analyzer import CourtCaseAnalyzer
from seo_ai_models.legal.court_practice.predictors.court_outcome_predictor import CourtOutcomePredictor
from seo_ai_models.legal.court_practice.models.court_case import (
    CourtCase, CaseAnalysis, CourtPracticeReport, CaseCategory, CaseStatus
)

# Создаем роутер
router = APIRouter(prefix="/court-practice", tags=["court-practice"])

# Инициализируем компоненты
court_parser = ArbitrageCourtParser()
case_analyzer = CourtCaseAnalyzer()
outcome_predictor = CourtOutcomePredictor()


# Модели запросов/ответов
class CourtSearchRequest(BaseModel):
    """Запрос на поиск судебных дел."""

    query: str = Field(..., description="Поисковый запрос")
    court_region: Optional[str] = Field(None, description="Регион суда")
    case_category: Optional[str] = Field(None, description="Категория дела")
    date_from: Optional[datetime] = Field(None, description="Дата начала периода")
    date_to: Optional[datetime] = Field(None, description="Дата окончания периода")
    max_results: int = Field(50, description="Максимальное количество результатов", ge=1, le=200)


class CaseAnalysisRequest(BaseModel):
    """Запрос на анализ судебного дела."""

    case_number: str = Field(..., description="Номер судебного дела")


class PracticeReportRequest(BaseModel):
    """Запрос на отчет по судебной практике."""

    query: str = Field(..., description="Тема для анализа практики")
    categories: Optional[List[str]] = Field(None, description="Категории дел для анализа")
    date_from: Optional[datetime] = Field(None, description="Начало периода анализа")
    date_to: Optional[datetime] = Field(None, description="Конец периода анализа")
    max_cases: int = Field(100, description="Максимальное количество дел для анализа", ge=10, le=500)


class CourtCaseResponse(BaseModel):
    """Ответ с информацией о судебном деле."""

    case_number: str
    court_name: str
    category: str
    status: str
    filing_date: Optional[datetime]
    claim_amount: Optional[float]
    description: str
    plaintiffs_count: int
    defendants_count: int
    decisions_count: int
    risk_level: str
    source_url: Optional[str]


class CaseAnalysisResponse(BaseModel):
    """Ответ с результатами анализа дела."""

    case: CourtCaseResponse
    semantic_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    predictive_outcome: Dict[str, Any]
    recommendations: List[str]
    similar_cases: List[Dict[str, Any]]
    analyzed_at: datetime


class PracticeReportResponse(BaseModel):
    """Ответ с отчетом по судебной практике."""

    query: str
    cases_found: int
    cases_analyzed: int
    category_distribution: Dict[str, int]
    outcome_statistics: Dict[str, int]
    average_decision_time: Optional[float]
    trends: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


@router.post("/search", response_model=List[CourtCaseResponse])
async def search_court_cases(
    request: CourtSearchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    analysis_id: Optional[str] = None
):
    """
    Поиск судебных дел по заданным критериям.
    """
    try:
        # Создаем ID анализа если не указан
        if not analysis_id:
            analysis_id = f"court_search_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(analysis_id, "Поиск судебных дел")

        # Выполняем поиск в фоне
        background_tasks.add_task(
            _perform_court_search,
            analysis_id,
            request
        )

        # Возвращаем начальный статус
        return await get_search_status(analysis_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска дел: {str(e)}")


@router.get("/search/status/{analysis_id}")
async def get_search_status(analysis_id: str):
    """
    Получение статуса поиска судебных дел.
    """
    try:
        status = progress_tracker.get_status(analysis_id)
        if not status:
            raise HTTPException(status_code=404, detail="Анализ не найден")

        if status.get("status") == "completed":
            # Возвращаем результаты
            results = status.get("results", [])
            return [
                CourtCaseResponse(
                    case_number=case.case_number,
                    court_name=case.court_name,
                    category=case.category.value,
                    status=case.status.value,
                    filing_date=case.filing_date,
                    claim_amount=case.claim_amount,
                    description=case.description[:200] + "..." if len(case.description) > 200 else case.description,
                    plaintiffs_count=len(case.plaintiffs),
                    defendants_count=len(case.defendants),
                    decisions_count=len(case.decisions),
                    risk_level=case.risk_level,
                    source_url=case.source_url
                ) for case in results
            ]
        else:
            # Возвращаем текущий статус
            return {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "current_step": status.get("current_step", ""),
                "message": status.get("message", "")
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.post("/analyze", response_model=CaseAnalysisResponse)
async def analyze_court_case(
    request: CaseAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    analysis_id: Optional[str] = None
):
    """
    Анализ конкретного судебного дела.
    """
    try:
        # Создаем ID анализа если не указан
        if not analysis_id:
            analysis_id = f"case_analysis_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(analysis_id, f"Анализ дела {request.case_number}")

        # Выполняем анализ в фоне
        background_tasks.add_task(
            _perform_case_analysis,
            analysis_id,
            request.case_number
        )

        # Возвращаем начальный статус
        return await get_analysis_status(analysis_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа дела: {str(e)}")


@router.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Получение статуса анализа судебного дела.
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

            return CaseAnalysisResponse(
                case=CourtCaseResponse(
                    case_number=analysis_result.case.case_number,
                    court_name=analysis_result.case.court_name,
                    category=analysis_result.case.category.value,
                    status=analysis_result.case.status.value,
                    filing_date=analysis_result.case.filing_date,
                    claim_amount=analysis_result.case.claim_amount,
                    description=analysis_result.case.description,
                    plaintiffs_count=len(analysis_result.case.plaintiffs),
                    defendants_count=len(analysis_result.case.defendants),
                    decisions_count=len(analysis_result.case.decisions),
                    risk_level=analysis_result.case.risk_level,
                    source_url=analysis_result.case.source_url
                ),
                semantic_analysis=analysis_result.semantic_analysis,
                risk_assessment=analysis_result.risk_assessment,
                predictive_outcome=analysis_result.predictive_outcome,
                recommendations=analysis_result.recommendations,
                similar_cases=analysis_result.similar_cases_analysis,
                analyzed_at=analysis_result.analyzed_at
            )
        else:
            # Возвращаем текущий статус
            return {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "current_step": status.get("current_step", ""),
                "message": status.get("message", "")
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса анализа: {str(e)}")


@router.post("/practice-report", response_model=PracticeReportResponse)
async def generate_practice_report(
    request: PracticeReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.ANALYST)),
    analysis_id: Optional[str] = None
):
    """
    Генерация отчета по судебной практике.
    """
    try:
        # Создаем ID анализа если не указан
        if not analysis_id:
            analysis_id = f"practice_report_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Начинаем отслеживание прогресса
        progress_tracker.start_tracking(analysis_id, f"Анализ практики: {request.query}")

        # Выполняем генерацию отчета в фоне
        background_tasks.add_task(
            _generate_practice_report_background,
            analysis_id,
            request
        )

        # Возвращаем начальный статус
        return await get_report_status(analysis_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации отчета: {str(e)}")


@router.get("/practice-report/status/{analysis_id}")
async def get_report_status(analysis_id: str):
    """
    Получение статуса генерации отчета по практике.
    """
    try:
        status = progress_tracker.get_status(analysis_id)
        if not status:
            raise HTTPException(status_code=404, detail="Анализ не найден")

        if status.get("status") == "completed":
            # Возвращаем результаты отчета
            report = status.get("results")
            if not report:
                raise HTTPException(status_code=500, detail="Результаты отчета не найдены")

            return PracticeReportResponse(
                query=report.query,
                cases_found=report.cases_found,
                cases_analyzed=report.cases_analyzed,
                category_distribution=report.category_distribution,
                outcome_statistics=report.outcome_statistics,
                average_decision_time=report.average_decision_time,
                trends=report.trends,
                recommendations=report.recommendations,
                generated_at=report.generated_at
            )
        else:
            # Возвращаем текущий статус
            return {
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "current_step": status.get("current_step", ""),
                "message": status.get("message", "")
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса отчета: {str(e)}")


# Вспомогательные функции для фоновой обработки

async def _perform_court_search(analysis_id: str, request: CourtSearchRequest):
    """
    Выполнение поиска судебных дел в фоне.
    """
    try:
        progress_tracker.update_progress(analysis_id, 10, "Подготовка параметров поиска")

        # Выполняем поиск
        progress_tracker.update_progress(analysis_id, 30, "Поиск дел в базе суда")

        cases = court_parser.search_cases(
            query=request.query,
            court_region=request.court_region,
            case_category=request.case_category,
            date_from=request.date_from,
            date_to=request.date_to,
            max_results=request.max_results
        )

        progress_tracker.update_progress(analysis_id, 90, "Обработка результатов")

        # Сохраняем результаты
        progress_tracker.complete_tracking(analysis_id, results=cases)

    except Exception as e:
        progress_tracker.fail_tracking(analysis_id, f"Ошибка поиска: {str(e)}")


async def _perform_case_analysis(analysis_id: str, case_number: str):
    """
    Выполнение анализа судебного дела в фоне.
    """
    try:
        progress_tracker.update_progress(analysis_id, 20, "Получение данных дела")

        # Получаем данные дела
        case = court_parser.get_case_details(case_number)
        if not case:
            progress_tracker.fail_tracking(analysis_id, f"Дело {case_number} не найдено")
            return

        progress_tracker.update_progress(analysis_id, 50, "Семантический анализ")

        # Выполняем анализ
        analysis_result = case_analyzer.analyze_case(case)

        progress_tracker.update_progress(analysis_id, 80, "Предиктивный анализ")

        # Добавляем предсказание исхода
        prediction = outcome_predictor.predict_outcome(case)
        analysis_result.predictive_outcome.update(prediction)

        progress_tracker.update_progress(analysis_id, 95, "Формирование отчета")

        # Сохраняем результаты
        progress_tracker.complete_tracking(analysis_id, results=analysis_result)

    except Exception as e:
        progress_tracker.fail_tracking(analysis_id, f"Ошибка анализа: {str(e)}")


async def _generate_practice_report_background(analysis_id: str, request: PracticeReportRequest):
    """
    Генерация отчета по судебной практике в фоне.
    """
    try:
        progress_tracker.update_progress(analysis_id, 15, "Сбор данных по практике")

        # Ищем дела по теме
        cases = court_parser.search_cases(
            query=request.query,
            max_results=request.max_cases
        )

        if not cases:
            progress_tracker.fail_tracking(analysis_id, "Не найдено дел по заданным критериям")
            return

        progress_tracker.update_progress(analysis_id, 40, f"Анализ {len(cases)} дел")

        # Анализируем каждое дело
        analyzed_cases = []
        for i, case in enumerate(cases):
            try:
                analysis = case_analyzer.analyze_case(case)
                analyzed_cases.append(analysis)

                # Обновляем прогресс
                progress = 40 + int(40 * (i + 1) / len(cases))
                progress_tracker.update_progress(
                    analysis_id, progress,
                    f"Анализ дела {i+1}/{len(cases)}"
                )

            except Exception as e:
                progress_tracker.update_progress(
                    analysis_id, progress,
                    f"Ошибка анализа дела {case.case_number}: {str(e)}"
                )
                continue

        progress_tracker.update_progress(analysis_id, 85, "Формирование статистики")

        # Формируем отчет
        report = _create_practice_report(request.query, analyzed_cases)

        progress_tracker.update_progress(analysis_id, 95, "Сохранение отчета")

        # Сохраняем результаты
        progress_tracker.complete_tracking(analysis_id, results=report)

    except Exception as e:
        progress_tracker.fail_tracking(analysis_id, f"Ошибка генерации отчета: {str(e)}")


def _create_practice_report(query: str, analyzed_cases: List[CaseAnalysis]) -> CourtPracticeReport:
    """
    Создание отчета по судебной практике на основе анализов.
    """
    try:
        # Собираем статистику
        category_distribution = {}
        outcome_statistics = {}
        decision_times = []

        for analysis in analyzed_cases:
            case = analysis.case

            # Распределение по категориям
            cat_key = case.category.value
            category_distribution[cat_key] = category_distribution.get(cat_key, 0) + 1

            # Статистика исходов
            predicted_outcome = analysis.predictive_outcome.get('predicted_outcome', 'неизвестен')
            outcome_statistics[predicted_outcome] = outcome_statistics.get(predicted_outcome, 0) + 1

            # Время принятия решения
            if case.filing_date and case.decisions:
                latest_decision = max(case.decisions, key=lambda d: d.date)
                decision_time = (latest_decision.date - case.filing_date).days
                decision_times.append(decision_time)

        # Среднее время принятия решения
        avg_decision_time = sum(decision_times) / len(decision_times) if decision_times else None

        # Анализ трендов (упрощенная версия)
        trends = {
            'most_common_category': max(category_distribution.items(), key=lambda x: x[1])[0] if category_distribution else None,
            'success_rate_trend': 'stable',  # В реальной реализации анализировать по времени
            'average_amount_trend': 'increasing'  # В реальной реализации анализировать динамику
        }

        # Рекомендации
        recommendations = []
        if avg_decision_time and avg_decision_time > 180:
            recommendations.append("Длительное рассмотрение дел - рассмотреть упрощенные процедуры")
        if len(analyzed_cases) > 0:
            success_rate = outcome_statistics.get('удовлетворен', 0) / len(analyzed_cases)
            if success_rate < 0.4:
                recommendations.append("Низкий процент удовлетворения исков - пересмотреть стратегию")

        return CourtPracticeReport(
            query=query,
            cases_found=len(analyzed_cases),
            cases_analyzed=len(analyzed_cases),
            category_distribution=category_distribution,
            outcome_statistics=outcome_statistics,
            average_decision_time=avg_decision_time,
            trends=trends,
            recommendations=recommendations
        )

    except Exception as e:
        # Возвращаем базовый отчет в случае ошибки
        return CourtPracticeReport(
            query=query,
            cases_found=len(analyzed_cases),
            cases_analyzed=0,
            recommendations=[f"Ошибка формирования отчета: {str(e)}"]
        )
