
"""
ReportGenerator - Модуль для создания отчетов и визуализаций по SEO-анализу.
Предоставляет функциональность для генерации подробных отчетов с различными
типами визуализаций на основе данных, полученных от анализаторов.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum
from pathlib import Path
import base64


class ReportType(Enum):
    """Типы отчетов."""
    OVERVIEW = "overview"
    CONTENT_ANALYSIS = "content_analysis"
    EEAT_ANALYSIS = "eeat_analysis"
    KEYWORD_ANALYSIS = "keyword_analysis"
    COMPETITORS_ANALYSIS = "competitors_analysis"
    LLM_COMPATIBILITY = "llm_compatibility"
    TECHNICAL_SEO = "technical_seo"
    CUSTOM = "custom"


class VisualizationType(Enum):
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


class Visualization:
    """Класс для визуализации данных в отчете."""
    
    def __init__(self,
                 title: str,
                 visualization_type: VisualizationType,
                 data: Dict[str, Any],
                 config: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        self.id = str(uuid4())
        self.title = title
        self.visualization_type = visualization_type
        self.data = data
        self.config = config or {}
        self.description = description
        self.created_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует визуализацию в словарь."""
        return {
            "id": self.id,
            "title": self.title,
            "visualization_type": self.visualization_type.value,
            "data": self.data,
            "config": self.config,
            "description": self.description,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Visualization':
        """Создает визуализацию из словаря."""
        visualization = cls(
            title=data.get("title", ""),
            visualization_type=VisualizationType(data.get("visualization_type", "table")),
            data=data.get("data", {}),
            config=data.get("config", {}),
            description=data.get("description", "")
        )
        
        visualization.id = data.get("id", str(uuid4()))
        visualization.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        
        return visualization


class Report:
    """Класс отчета по SEO-анализу."""
    
    def __init__(self,
                 title: str,
                 project_id: str,
                 report_type: ReportType,
                 description: str = "",
                 created_by: Optional[str] = None,
                 created_at: Optional[datetime] = None):
        self.id = str(uuid4())
        self.title = title
        self.project_id = project_id
        self.report_type = report_type
        self.description = description
        self.created_by = created_by
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at
        self.visualizations: List[Visualization] = []
        self.metadata = {}
        self.content = {}
        
    def add_visualization(self, visualization: Visualization):
        """Добавляет визуализацию в отчет."""
        self.visualizations.append(visualization)
        self.updated_at = datetime.now()
        
    def remove_visualization(self, visualization_id: str) -> bool:
        """Удаляет визуализацию из отчета."""
        for i, vis in enumerate(self.visualizations):
            if vis.id == visualization_id:
                self.visualizations.pop(i)
                self.updated_at = datetime.now()
                return True
        return False
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует отчет в словарь."""
        return {
            "id": self.id,
            "title": self.title,
            "project_id": self.project_id,
            "report_type": self.report_type.value,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "visualizations": [vis.to_dict() for vis in self.visualizations],
            "metadata": self.metadata,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Report':
        """Создает отчет из словаря."""
        report = cls(
            title=data.get("title", ""),
            project_id=data.get("project_id", ""),
            report_type=ReportType(data.get("report_type", "overview")),
            description=data.get("description", ""),
            created_by=data.get("created_by")
        )
        
        report.id = data.get("id", str(uuid4()))
        report.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        report.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        report.metadata = data.get("metadata", {})
        report.content = data.get("content", {})
        
        # Загрузка визуализаций
        for vis_data in data.get("visualizations", []):
            report.visualizations.append(Visualization.from_dict(vis_data))
            
        return report


class ReportGenerator:
    """Генератор отчетов для создания и управления отчетами."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./data/reports")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports: Dict[str, Report] = {}
        
    def load_reports(self):
        """Загружает отчеты из файлов."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for report_file in self.data_dir.glob("*.json"):
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                report = Report.from_dict(report_data)
                self.reports[report.id] = report
            except Exception as e:
                logging.error(f"Failed to load report from {report_file}: {str(e)}")
                
    def create_report(self, title: str, project_id: str, report_type: ReportType, **kwargs) -> Report:
        """Создает новый отчет."""
        report = Report(
            title=title,
            project_id=project_id,
            report_type=report_type,
            **kwargs
        )
        self.reports[report.id] = report
        self._save_report(report)
        return report
        
    def update_report(self, report_id: str, **kwargs) -> Optional[Report]:
        """Обновляет существующий отчет."""
        if report_id not in self.reports:
            return None
            
        report = self.reports[report_id]
        for key, value in kwargs.items():
            if hasattr(report, key):
                setattr(report, key, value)
                
        report.updated_at = datetime.now()
        self._save_report(report)
        return report
        
    def add_visualization(self, report_id: str, 
                          title: str, 
                          visualization_type: VisualizationType,
                          data: Dict[str, Any],
                          **kwargs) -> Optional[Visualization]:
        """Добавляет визуализацию в отчет."""
        if report_id not in self.reports:
            return None
            
        visualization = Visualization(
            title=title,
            visualization_type=visualization_type,
            data=data,
            **kwargs
        )
        
        report = self.reports[report_id]
        report.add_visualization(visualization)
        self._save_report(report)
        return visualization
        
    def remove_visualization(self, report_id: str, visualization_id: str) -> bool:
        """Удаляет визуализацию из отчета."""
        if report_id not in self.reports:
            return False
            
        report = self.reports[report_id]
        result = report.remove_visualization(visualization_id)
        if result:
            self._save_report(report)
        return result
        
    def _save_report(self, report: Report):
        """Сохраняет отчет в файл."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        file_path = self.data_dir / f"{report.id}.json"
        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
            
    def get_report(self, report_id: str) -> Optional[Report]:
        """Возвращает отчет по ID."""
        return self.reports.get(report_id)
        
    def get_project_reports(self, project_id: str) -> List[Report]:
        """Возвращает список отчетов проекта."""
        return [report for report in self.reports.values() if report.project_id == project_id]
        
    def delete_report(self, report_id: str) -> bool:
        """Удаляет отчет."""
        if report_id not in self.reports:
            return False
            
        del self.reports[report_id]
        report_file = self.data_dir / f"{report_id}.json"
        if report_file.exists():
            report_file.unlink()
            
        return True
        
    def export_report_to_html(self, report_id: str) -> Optional[str]:
        """Экспортирует отчет в HTML формат."""
        if report_id not in self.reports:
            return None
            
        report = self.reports[report_id]
        # Здесь будет код для генерации HTML
        # Пока просто заглушка
        html_content = f"<html><head><title>{report.title}</title></head><body><h1>{report.title}</h1></body></html>"
        return html_content
        
    def export_report_to_pdf(self, report_id: str) -> Optional[bytes]:
        """Экспортирует отчет в PDF формат."""
        if report_id not in self.reports:
            return None
            
        # Здесь будет код для генерации PDF
        # Пока просто заглушка
        return b"PDF content"


# Функция для создания экземпляра ReportGenerator
def create_report_generator(data_dir: Optional[str] = None) -> ReportGenerator:
    """Создает экземпляр генератора отчетов."""
    generator = ReportGenerator(data_dir)
    generator.load_reports()
    return generator
