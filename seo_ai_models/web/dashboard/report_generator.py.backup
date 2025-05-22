
"""
ReportGenerator - Модуль для генерации отчетов для панели управления.
Обеспечивает функциональность создания и управления отчетами на основе анализов.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
import os
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class Report:
    """Класс, представляющий отчет в системе."""
    
    def __init__(self,
                 report_id: str,
                 name: str,
                 created_at: Optional[datetime] = None,
                 updated_at: Optional[datetime] = None,
                 owner_id: Optional[str] = None,
                 description: str = "",
                 project_id: Optional[str] = None,
                 analysis_id: Optional[str] = None,
                 type: str = "analysis",
                 data: Optional[Dict[str, Any]] = None,
                 status: str = "draft"):
        """
        Инициализирует отчет.
        
        Args:
            report_id: Уникальный идентификатор отчета
            name: Название отчета
            created_at: Время создания отчета
            updated_at: Время последнего обновления отчета
            owner_id: ID владельца отчета
            description: Описание отчета
            project_id: ID проекта, к которому относится отчет
            analysis_id: ID анализа, на основе которого создан отчет
            type: Тип отчета (analysis, comparison, summary, custom)
            data: Данные отчета
            status: Статус отчета (draft, published, archived)
        """
        self.report_id = report_id
        self.name = name
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.owner_id = owner_id
        self.description = description
        self.project_id = project_id
        self.analysis_id = analysis_id
        self.type = type
        self.data = data or {}
        self.status = status
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует отчет в словарь."""
        return {
            "report_id": self.report_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_id": self.owner_id,
            "description": self.description,
            "project_id": self.project_id,
            "analysis_id": self.analysis_id,
            "type": self.type,
            "data": self.data,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Report':
        """Создает отчет из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at")
        updated_at = datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at")
        
        return cls(
            report_id=data["report_id"],
            name=data["name"],
            created_at=created_at,
            updated_at=updated_at,
            owner_id=data.get("owner_id"),
            description=data.get("description", ""),
            project_id=data.get("project_id"),
            analysis_id=data.get("analysis_id"),
            type=data.get("type", "analysis"),
            data=data.get("data", {}),
            status=data.get("status", "draft")
        )


class ReportTemplate:
    """Класс, представляющий шаблон отчета."""
    
    def __init__(self,
                 template_id: str,
                 name: str,
                 description: str = "",
                 type: str = "analysis",
                 structure: Optional[Dict[str, Any]] = None,
                 created_at: Optional[datetime] = None,
                 updated_at: Optional[datetime] = None,
                 creator_id: Optional[str] = None):
        """
        Инициализирует шаблон отчета.
        
        Args:
            template_id: Уникальный идентификатор шаблона
            name: Название шаблона
            description: Описание шаблона
            type: Тип отчета, для которого предназначен шаблон
            structure: Структура отчета
            created_at: Время создания шаблона
            updated_at: Время последнего обновления шаблона
            creator_id: ID создателя шаблона
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.type = type
        self.structure = structure or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.creator_id = creator_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует шаблон в словарь."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "structure": self.structure,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "creator_id": self.creator_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportTemplate':
        """Создает шаблон из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at")
        updated_at = datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at")
        
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data.get("description", ""),
            type=data.get("type", "analysis"),
            structure=data.get("structure", {}),
            created_at=created_at,
            updated_at=updated_at,
            creator_id=data.get("creator_id")
        )


class ReportGenerator:
    """
    Класс для генерации и управления отчетами.
    """
    
    def __init__(self, data_dir: Optional[str] = None, api_client=None, project_management=None):
        """
        Инициализирует генератор отчетов.
        
        Args:
            data_dir: Директория для хранения данных отчетов (для локального режима)
            api_client: Клиент API для взаимодействия с бэкендом
            project_management: Объект управления проектами для доступа к анализам
        """
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".seo_ai_models", "reports")
        self.api_client = api_client
        self.project_management = project_management
        
        self.reports = {}
        self.templates = {}
        
        # Создаем директорию для данных, если она не существует
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Загружаем существующие отчеты и шаблоны
        self._load_reports()
        self._load_templates()
        
        # Создаем стандартные шаблоны, если они отсутствуют
        if not self.templates:
            self._create_default_templates()
    
    def _load_reports(self):
        """Загружает существующие отчеты из хранилища."""
        reports_dir = os.path.join(self.data_dir, "reports")
        
        # Создаем директорию, если она не существует
        os.makedirs(reports_dir, exist_ok=True)
        
        # Загружаем отчеты
        for report_file in Path(reports_dir).glob("*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    report = Report.from_dict(report_data)
                    self.reports[report.report_id] = report
            except Exception as e:
                logger.error(f"Failed to load report from {report_file}: {str(e)}")
    
    def _load_templates(self):
        """Загружает существующие шаблоны отчетов из хранилища."""
        templates_dir = os.path.join(self.data_dir, "templates")
        
        # Создаем директорию, если она не существует
        os.makedirs(templates_dir, exist_ok=True)
        
        # Загружаем шаблоны
        for template_file in Path(templates_dir).glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    template = ReportTemplate.from_dict(template_data)
                    self.templates[template.template_id] = template
            except Exception as e:
                logger.error(f"Failed to load template from {template_file}: {str(e)}")
    
    def _create_default_templates(self):
        """Создает стандартные шаблоны отчетов."""
        # Шаблон для анализа проекта
        analysis_template = ReportTemplate(
            template_id=str(uuid.uuid4()),
            name="Стандартный анализ проекта",
            description="Стандартный шаблон для отчета по анализу проекта",
            type="analysis",
            structure={
                "sections": [
                    {
                        "id": "summary",
                        "title": "Сводка",
                        "description": "Общая сводка по результатам анализа",
                        "content_type": "summary"
                    },
                    {
                        "id": "content_analysis",
                        "title": "Анализ контента",
                        "description": "Подробный анализ контента",
                        "content_type": "content_metrics"
                    },
                    {
                        "id": "eeat_analysis",
                        "title": "Анализ E-E-A-T",
                        "description": "Анализ E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)",
                        "content_type": "eeat_metrics"
                    },
                    {
                        "id": "llm_compatibility",
                        "title": "Совместимость с LLM",
                        "description": "Анализ совместимости контента с требованиями LLM",
                        "content_type": "llm_metrics"
                    },
                    {
                        "id": "recommendations",
                        "title": "Рекомендации",
                        "description": "Рекомендации по улучшению",
                        "content_type": "recommendations"
                    }
                ],
                "charts": [
                    {
                        "id": "content_metrics_chart",
                        "title": "Метрики контента",
                        "type": "radar",
                        "data_source": "content_metrics"
                    },
                    {
                        "id": "eeat_metrics_chart",
                        "title": "Метрики E-E-A-T",
                        "type": "radar",
                        "data_source": "eeat_metrics"
                    },
                    {
                        "id": "llm_metrics_chart",
                        "title": "Метрики LLM",
                        "type": "radar",
                        "data_source": "llm_metrics"
                    }
                ]
            }
        )
        
        # Шаблон для сравнения проектов
        comparison_template = ReportTemplate(
            template_id=str(uuid.uuid4()),
            name="Сравнение проектов",
            description="Шаблон для сравнения нескольких проектов",
            type="comparison",
            structure={
                "sections": [
                    {
                        "id": "summary",
                        "title": "Сводка",
                        "description": "Общая сводка по результатам сравнения",
                        "content_type": "summary"
                    },
                    {
                        "id": "content_comparison",
                        "title": "Сравнение контента",
                        "description": "Сравнение контента проектов",
                        "content_type": "content_comparison"
                    },
                    {
                        "id": "eeat_comparison",
                        "title": "Сравнение E-E-A-T",
                        "description": "Сравнение E-E-A-T проектов",
                        "content_type": "eeat_comparison"
                    },
                    {
                        "id": "llm_comparison",
                        "title": "Сравнение совместимости с LLM",
                        "description": "Сравнение совместимости проектов с требованиями LLM",
                        "content_type": "llm_comparison"
                    },
                    {
                        "id": "recommendations",
                        "title": "Рекомендации",
                        "description": "Сравнительные рекомендации по улучшению",
                        "content_type": "comparison_recommendations"
                    }
                ],
                "charts": [
                    {
                        "id": "content_comparison_chart",
                        "title": "Сравнение метрик контента",
                        "type": "bar",
                        "data_source": "content_comparison"
                    },
                    {
                        "id": "eeat_comparison_chart",
                        "title": "Сравнение метрик E-E-A-T",
                        "type": "bar",
                        "data_source": "eeat_comparison"
                    },
                    {
                        "id": "llm_comparison_chart",
                        "title": "Сравнение метрик LLM",
                        "type": "bar",
                        "data_source": "llm_comparison"
                    }
                ]
            }
        )
        
        # Шаблон для LLM-оптимизации
        llm_template = ReportTemplate(
            template_id=str(uuid.uuid4()),
            name="LLM-оптимизация",
            description="Шаблон для отчета по LLM-оптимизации",
            type="llm_optimization",
            structure={
                "sections": [
                    {
                        "id": "summary",
                        "title": "Сводка",
                        "description": "Общая сводка по результатам анализа LLM-совместимости",
                        "content_type": "summary"
                    },
                    {
                        "id": "llm_content_analysis",
                        "title": "Анализ контента для LLM",
                        "description": "Подробный анализ контента с точки зрения LLM",
                        "content_type": "llm_content_metrics"
                    },
                    {
                        "id": "citability_analysis",
                        "title": "Анализ цитируемости",
                        "description": "Анализ потенциала цитирования контента в LLM-ответах",
                        "content_type": "citability_metrics"
                    },
                    {
                        "id": "semantic_structure",
                        "title": "Семантическая структура",
                        "description": "Анализ семантической структуры контента",
                        "content_type": "semantic_metrics"
                    },
                    {
                        "id": "llm_recommendations",
                        "title": "Рекомендации по LLM-оптимизации",
                        "description": "Рекомендации по улучшению для LLM",
                        "content_type": "llm_recommendations"
                    }
                ],
                "charts": [
                    {
                        "id": "llm_metrics_chart",
                        "title": "Метрики LLM",
                        "type": "radar",
                        "data_source": "llm_metrics"
                    },
                    {
                        "id": "citability_chart",
                        "title": "Метрики цитируемости",
                        "type": "bar",
                        "data_source": "citability_metrics"
                    },
                    {
                        "id": "semantic_structure_chart",
                        "title": "Семантическая структура",
                        "type": "tree",
                        "data_source": "semantic_metrics"
                    }
                ]
            }
        )
        
        # Сохраняем шаблоны
        self.templates[analysis_template.template_id] = analysis_template
        self.templates[comparison_template.template_id] = comparison_template
        self.templates[llm_template.template_id] = llm_template
        
        self._save_template(analysis_template)
        self._save_template(comparison_template)
        self._save_template(llm_template)
    
    def _save_template(self, template: ReportTemplate):
        """
        Сохраняет шаблон отчета в хранилище.
        
        Args:
            template: Шаблон для сохранения
        """
        templates_dir = os.path.join(self.data_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        template_file = os.path.join(templates_dir, f"{template.template_id}.json")
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)
    
    def create_report(self, name: str, template_id: str,
                     project_id: Optional[str] = None,
                     analysis_id: Optional[str] = None,
                     description: str = "",
                     owner_id: Optional[str] = None,
                     data: Optional[Dict[str, Any]] = None) -> Report:
        """
        Создает новый отчет.
        
        Args:
            name: Название отчета
            template_id: ID шаблона
            project_id: ID проекта
            analysis_id: ID анализа
            description: Описание отчета
            owner_id: ID владельца отчета
            data: Данные отчета
            
        Returns:
            Report: Созданный отчет
        """
        # Получаем шаблон
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template with ID {template_id} not found")
        
        # Генерируем уникальный ID для отчета
        report_id = str(uuid.uuid4())
        
        # Создаем отчет
        report = Report(
            report_id=report_id,
            name=name,
            description=description,
            project_id=project_id,
            analysis_id=analysis_id,
            type=template.type,
            owner_id=owner_id,
            data=data or {
                "template_id": template_id,
                "template_structure": template.structure,
                "content": {}
            }
        )
        
        # Сохраняем отчет
        self.reports[report_id] = report
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: Report):
        """
        Сохраняет отчет в хранилище.
        
        Args:
            report: Отчет для сохранения
        """
        reports_dir = os.path.join(self.data_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = os.path.join(reports_dir, f"{report.report_id}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """
        Получает отчет по ID.
        
        Args:
            report_id: ID отчета
            
        Returns:
            Optional[Report]: Отчет, если найден, иначе None
        """
        return self.reports.get(report_id)
    
    def get_reports(self, owner_id: Optional[str] = None,
                  project_id: Optional[str] = None,
                  analysis_id: Optional[str] = None,
                  type: Optional[str] = None,
                  status: Optional[str] = None) -> List[Report]:
        """
        Получает список отчетов с возможностью фильтрации.
        
        Args:
            owner_id: Фильтр по ID владельца
            project_id: Фильтр по ID проекта
            analysis_id: Фильтр по ID анализа
            type: Фильтр по типу отчета
            status: Фильтр по статусу
            
        Returns:
            List[Report]: Список отчетов
        """
        result = []
        
        for report in self.reports.values():
            if owner_id and report.owner_id != owner_id:
                continue
            if project_id and report.project_id != project_id:
                continue
            if analysis_id and report.analysis_id != analysis_id:
                continue
            if type and report.type != type:
                continue
            if status and report.status != status:
                continue
            result.append(report)
        
        return result
    
    def update_report(self, report_id: str,
                     name: Optional[str] = None,
                     description: Optional[str] = None,
                     data: Optional[Dict[str, Any]] = None,
                     status: Optional[str] = None) -> Optional[Report]:
        """
        Обновляет отчет.
        
        Args:
            report_id: ID отчета
            name: Новое название отчета
            description: Новое описание
            data: Новые данные отчета
            status: Новый статус
            
        Returns:
            Optional[Report]: Обновленный отчет, если найден, иначе None
        """
        report = self.get_report(report_id)
        if not report:
            return None
        
        if name:
            report.name = name
        if description:
            report.description = description
        if data:
            report.data.update(data)
        if status:
            report.status = status
        
        report.updated_at = datetime.now()
        
        # Сохраняем отчет
        self._save_report(report)
        
        return report
    
    def delete_report(self, report_id: str) -> bool:
        """
        Удаляет отчет.
        
        Args:
            report_id: ID отчета
            
        Returns:
            bool: True, если отчет успешно удален, иначе False
        """
        report = self.get_report(report_id)
        if not report:
            return False
        
        # Помечаем отчет как архивный
        report.status = "archived"
        report.updated_at = datetime.now()
        
        # Сохраняем отчет
        self._save_report(report)
        
        return True
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """
        Получает шаблон отчета по ID.
        
        Args:
            template_id: ID шаблона
            
        Returns:
            Optional[ReportTemplate]: Шаблон, если найден, иначе None
        """
        return self.templates.get(template_id)
    
    def get_templates(self, type: Optional[str] = None) -> List[ReportTemplate]:
        """
        Получает список шаблонов отчетов с возможностью фильтрации.
        
        Args:
            type: Фильтр по типу отчета
            
        Returns:
            List[ReportTemplate]: Список шаблонов
        """
        result = []
        
        for template in self.templates.values():
            if type and template.type != type:
                continue
            result.append(template)
        
        return result
    
    def create_template(self, name: str, type: str,
                       description: str = "",
                       structure: Optional[Dict[str, Any]] = None,
                       creator_id: Optional[str] = None) -> ReportTemplate:
        """
        Создает новый шаблон отчета.
        
        Args:
            name: Название шаблона
            type: Тип отчета
            description: Описание шаблона
            structure: Структура отчета
            creator_id: ID создателя шаблона
            
        Returns:
            ReportTemplate: Созданный шаблон
        """
        # Генерируем уникальный ID для шаблона
        template_id = str(uuid.uuid4())
        
        # Создаем шаблон
        template = ReportTemplate(
            template_id=template_id,
            name=name,
            description=description,
            type=type,
            structure=structure or {},
            creator_id=creator_id
        )
        
        # Сохраняем шаблон
        self.templates[template_id] = template
        self._save_template(template)
        
        return template
    
    def update_template(self, template_id: str,
                       name: Optional[str] = None,
                       description: Optional[str] = None,
                       structure: Optional[Dict[str, Any]] = None) -> Optional[ReportTemplate]:
        """
        Обновляет шаблон отчета.
        
        Args:
            template_id: ID шаблона
            name: Новое название шаблона
            description: Новое описание
            structure: Новая структура отчета
            
        Returns:
            Optional[ReportTemplate]: Обновленный шаблон, если найден, иначе None
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        if name:
            template.name = name
        if description:
            template.description = description
        if structure:
            template.structure = structure
        
        template.updated_at = datetime.now()
        
        # Сохраняем шаблон
        self._save_template(template)
        
        return template
    
    def delete_template(self, template_id: str) -> bool:
        """
        Удаляет шаблон отчета.
        
        Args:
            template_id: ID шаблона
            
        Returns:
            bool: True, если шаблон успешно удален, иначе False
        """
        if template_id not in self.templates:
            return False
        
        # Удаляем шаблон
        del self.templates[template_id]
        
        # Удаляем файл шаблона
        template_file = os.path.join(self.data_dir, "templates", f"{template_id}.json")
        if os.path.exists(template_file):
            os.remove(template_file)
        
        return True
    
    def generate_report_from_analysis(self, analysis_id: str, template_id: str,
                                     name: Optional[str] = None,
                                     description: Optional[str] = None,
                                     owner_id: Optional[str] = None) -> Optional[Report]:
        """
        Генерирует отчет на основе анализа.
        
        Args:
            analysis_id: ID анализа
            template_id: ID шаблона
            name: Название отчета
            description: Описание отчета
            owner_id: ID владельца отчета
            
        Returns:
            Optional[Report]: Созданный отчет, если анализ найден, иначе None
        """
        if not self.project_management:
            raise ValueError("ProjectManagement is required for generating reports from analyses")
        
        # Получаем анализ
        analysis = self.project_management.get_analysis(analysis_id)
        if not analysis:
            return None
        
        # Получаем проект
        project = self.project_management.get_project(analysis.project_id)
        if not project:
            return None
        
        # Получаем шаблон
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template with ID {template_id} not found")
        
        # Формируем название отчета, если не указано
        if not name:
            name = f"Отчет по проекту '{project.name}' ({datetime.now().strftime('%d.%m.%Y')})"
        
        # Формируем описание отчета, если не указано
        if not description:
            description = f"Отчет по результатам анализа проекта '{project.name}'"
        
        # Обрабатываем данные анализа в соответствии с шаблоном
        data = self._process_analysis_data(analysis, template)
        
        # Создаем отчет
        report = self.create_report(
            name=name,
            template_id=template_id,
            project_id=project.project_id,
            analysis_id=analysis_id,
            description=description,
            owner_id=owner_id,
            data=data
        )
        
        return report
    
    def _process_analysis_data(self, analysis, template: ReportTemplate) -> Dict[str, Any]:
        """
        Обрабатывает данные анализа в соответствии с шаблоном.
        
        Args:
            analysis: Анализ
            template: Шаблон отчета
            
        Returns:
            Dict[str, Any]: Обработанные данные для отчета
        """
        # Здесь должна быть логика обработки данных анализа
        # в соответствии с шаблоном
        
        # Пока просто создаем структуру на основе шаблона
        data = {
            "template_id": template.template_id,
            "template_structure": template.structure,
            "content": {}
        }
        
        # Заполняем контент на основе результатов анализа
        if analysis.results:
            # Для каждой секции в шаблоне
            for section in template.structure.get("sections", []):
                section_id = section["id"]
                content_type = section.get("content_type")
                
                # Заполняем контент в зависимости от типа
                if content_type == "summary":
                    data["content"][section_id] = {
                        "summary_text": analysis.results.get("summary", "Сводка по анализу отсутствует"),
                        "overall_score": analysis.results.get("overall_score", 0)
                    }
                elif content_type == "content_metrics":
                    data["content"][section_id] = analysis.results.get("content_analysis", {})
                elif content_type == "eeat_metrics":
                    data["content"][section_id] = analysis.results.get("eeat_analysis", {})
                elif content_type == "llm_metrics":
                    data["content"][section_id] = analysis.results.get("llm_analysis", {})
                elif content_type == "recommendations":
                    data["content"][section_id] = {
                        "recommendations": analysis.results.get("recommendations", [])
                    }
        
        return data
    
    def get_recent_reports(self, limit: int = 10) -> List[Report]:
        """
        Получает список последних отчетов.
        
        Args:
            limit: Максимальное количество отчетов
            
        Returns:
            List[Report]: Список отчетов
        """
        # Сортируем отчеты по дате обновления (от новых к старым)
        sorted_reports = sorted(
            self.reports.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )
        
        return sorted_reports[:limit]
    
    def get_reports_by_status(self, status: str, limit: int = 100) -> List[Report]:
        """
        Получает список отчетов по статусу.
        
        Args:
            status: Статус отчетов
            limit: Максимальное количество отчетов
            
        Returns:
            List[Report]: Список отчетов
        """
        # Фильтруем отчеты по статусу и сортируем по дате обновления
        filtered_reports = sorted(
            [r for r in self.reports.values() if r.status == status],
            key=lambda x: x.updated_at,
            reverse=True
        )
        
        return filtered_reports[:limit]
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по отчетам.
        
        Returns:
            Dict[str, Any]: Статистика по отчетам
        """
        draft_reports = len([r for r in self.reports.values() if r.status == "draft"])
        published_reports = len([r for r in self.reports.values() if r.status == "published"])
        archived_reports = len([r for r in self.reports.values() if r.status == "archived"])
        
        return {
            "total_reports": len(self.reports),
            "draft_reports": draft_reports,
            "published_reports": published_reports,
            "archived_reports": archived_reports,
            "total_templates": len(self.templates)
        }
