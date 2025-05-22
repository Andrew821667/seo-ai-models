
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем заглушку в update_report
content = content.replace(
    """        if settings:
            report.settings.update(settings)
        
        report.updated_at = datetime.now()
        
        # Сохраняем отчет
        self._save_report(report)
        return None""",
    """        if settings:
            report.settings.update(settings)
        
        report.updated_at = datetime.now()
        
        # Сохраняем отчет
        self._save_report(report)
        
        print(f"✅ Отчет {report_id} успешно обновлен")
        return report"""
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка #1 (update_report) исправлена!")
