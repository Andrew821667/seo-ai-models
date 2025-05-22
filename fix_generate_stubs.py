
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем заглушку после проверки анализа
content = content.replace(
    """    if not analysis:
        return None""",
    """    if not analysis:
        print(f"❌ Анализ {analysis_id} не найден")
        return None"""
)

# Заменяем заглушку после проверки проекта
content = content.replace(
    """    if not project:
        return None""",
    """    if not project:
        print(f"❌ Проект {analysis.project_id} не найден для анализа {analysis_id}")
        return None"""
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушки #3 и #4 исправлены!")
