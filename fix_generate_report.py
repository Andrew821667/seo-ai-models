
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем заглушку после "if not analysis:"
content = content.replace(
    """    # Получаем анализ
    analysis = self.project_management.get_analysis(analysis_id)
    if not analysis:
        return None""",
    """    # Получаем анализ
    analysis = self.project_management.get_analysis(analysis_id)
    if not analysis:
        print(f"❌ Анализ {analysis_id} не найден")
        return None"""
)

# Заменяем заглушку после "if not project:"
content = content.replace(
    """    # Получаем проект
    project = self.project_management.get_project(analysis.project_id)
    if not project:
        return None""",
    """    # Получаем проект
    project = self.project_management.get_project(analysis.project_id)
    if not project:
        print(f"❌ Проект {analysis.project_id} не найден для анализа {analysis_id}")
        return None"""
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушки #3 и #4 (generate_report_from_analysis) исправлены!")
