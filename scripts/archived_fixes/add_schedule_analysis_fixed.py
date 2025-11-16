
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Правильно отформатированный метод schedule_analysis
new_method = """
    def schedule_analysis(self, project_id: str, analysis_name: str, 
                         analysis_type: str = 'full_seo',
                         scheduled_time: Optional[datetime] = None,
                         priority: str = 'normal') -> Optional[Analysis]:
        """
        Планирует выполнение анализа проекта.
        
        Args:
            project_id: ID проекта
            analysis_name: Название анализа
            analysis_type: Тип анализа (full_seo, quick_scan, content_audit)
            scheduled_time: Время запланированного выполнения
            priority: Приоритет выполнения (low, normal, high, urgent)
            
        Returns:
            Optional[Analysis]: Запланированный анализ, если проект найден, иначе None
        """
        from datetime import timedelta
        
        project = self.get_project(project_id)
        if not project:
            print(f'❌ Проект {project_id} не найден для планирования анализа')
            return None
            
        # Генерируем уникальный ID
        import uuid
        analysis_id = str(uuid.uuid4())
        
        # Устанавливаем время выполнения
        if not scheduled_time:
            scheduled_time = datetime.now() + timedelta(minutes=5)
            
        # Создаем запланированный анализ
        analysis = Analysis(
            analysis_id=analysis_id,
            project_id=project_id,
            name=analysis_name,
            type=analysis_type,
            status='scheduled',
            created_at=datetime.now(),
            settings={
                'scheduled_time': scheduled_time.isoformat(),
                'priority': priority
            }
        )
        
        # Сохраняем анализ
        self.analyses[analysis_id] = analysis
        project.analyses.append(analysis_id)
        self._save_analysis(analysis)
        
        print(f'✅ Анализ \'{analysis_name}\' запланирован на {scheduled_time.strftime("%Y-%m-%d %H:%M:%S")}')
        return analysis

"""

# Вставляем перед методом _save_analysis
content = content.replace(
    "    def _save_analysis(self, analysis: Analysis):",
    new_method + "    def _save_analysis(self, analysis: Analysis):"
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Метод schedule_analysis добавлен с правильными отступами!")
