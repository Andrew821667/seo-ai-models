
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Найдем заглушку в update_analysis_status
for i, line in enumerate(lines):
    if i >= 460 and i <= 470 and "return None" in line.strip() and line.strip() == "return None":
        # Проверим, что мы в методе update_analysis_status
        method_found = False
        for j in range(max(0, i-20), i):
            if "def update_analysis_status" in lines[j]:
                method_found = True
                break
        
        if method_found:
            # Заменяем заглушку
            lines[i] = """        # Сохраняем обновленный анализ
        self._save_analysis(analysis)
        
        print(f"✅ Статус анализа {analysis_id} обновлен на '{status}'")
        
        # Дополнительные уведомления
        if status == "completed":
            print(f"🎉 Анализ {analysis_id} успешно завершен!")
        elif status == "failed":
            print(f"❌ Анализ {analysis_id} завершился с ошибкой")
            
        return analysis
"""
            break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Заглушка #3 (update_analysis_status) исправлена!")
