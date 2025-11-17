"""
Модели данных для отчетов и визуализаций.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ReportType(str, Enum):
    """Типы отчетов."""

    OVERVIEW = "overview"
    CONTENT_ANALYSIS = "content_analysis"
    EEAT_ANALYSIS = "eeat_analysis"
    KEYWORD_ANALYSIS = "keyword_analysis"
    COMPETITORS_ANALYSIS = "competitors_analysis"
    LLM_COMPATIBILITY = "llm_compatibility"
    TECHNICAL_SEO = "technical_seo"
    CUSTOM = "custom"


class VisualizationType(str, Enum):
    """Типы визуализаций."""

    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    RADAR_CHART = "radar_chart"
    HEATMAP = "heatmap"
    TABLE = "table"
    SCORE_CARD = "score_card"
    METRICS_GRID = "metrics_grid"
    TIMELINE = "timeline"
    CUSTOM = "custom"


class VisualizationCreate(BaseModel):
    """Модель для создания визуализации."""

    title: str = Field(..., min_length=3, max_length=100)
    visualization_type: VisualizationType
    data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class VisualizationResponse(BaseModel):
    """Модель ответа с данными визуализации."""

    id: str
    title: str
    visualization_type: VisualizationType
    data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: datetime


class ReportCreate(BaseModel):
    """Модель для создания отчета."""

    title: str = Field(..., min_length=3, max_length=100)
    report_type: ReportType
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[VisualizationCreate]] = None


class ReportResponse(BaseModel):
    """Модель ответа с данными отчета."""

    id: str
    title: str
    project_id: str
    report_type: ReportType
    description: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    visualizations: List[VisualizationResponse] = []
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[Dict[str, Any]] = None


class ReportUpdate(BaseModel):
    """Модель для обновления отчета."""

    title: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[Dict[str, Any]] = None
