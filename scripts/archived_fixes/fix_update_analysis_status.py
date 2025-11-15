
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменим заглушку в update_analysis_status
content = content.replace(
    """        # Сохраняем анализ
        self._save_analysis(analysis)
        return None""",
    """        # Сохраняем анализ
        self._save_analysis(analysis)
        
        print(f"✅ Статус анализа {analysis_id} обновлен на '{status}'")
        if status == "completed":
            print(f"🎉 Анализ {analysis_id} успешно завершен!")
        elif status == "failed":
            print(f"❌ Анализ {analysis_id} завершился с ошибкой")
            
        return analysis"""
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка в update_analysis_status исправлена!")
