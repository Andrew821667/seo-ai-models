
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем все return None на правильные возвраты
# Метод update_report
content = content.replace(
    """        self._save_report(report)
        return None""",
    """        self._save_report(report)
        print(f"✅ Отчет {report_id} успешно обновлен")
        return report"""
)

# Метод update_template  
content = content.replace(
    """        self._save_template(template)
        return None""",
    """        self._save_template(template)
        print(f"✅ Шаблон {template_id} успешно обновлен")
        return template"""
)

# Метод generate_report_from_analysis - первая заглушка
content = content.replace(
    """    if not analysis:
        return None""",
    """    if not analysis:
        print(f"❌ Анализ {analysis_id} не найден")
        return None"""
)

# Метод generate_report_from_analysis - вторая заглушка
content = content.replace(
    """    if not project:
        return None""",
    """    if not project:
        print(f"❌ Проект для анализа {analysis_id} не найден")
        return None"""
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все заглушки исправлены!")
