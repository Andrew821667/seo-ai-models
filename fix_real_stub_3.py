
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

# Находим заглушку в update_analysis_status
for i, line in enumerate(lines):
    if (i >= 448 and i <= 452 and 
        "return None" in line and
        any("def update_analysis_status" in lines[j] for j in range(max(0, i-20), i))):
        
        lines[i] = """        # Сохраняем обновленный анализ
        self._save_analysis(analysis)
        
        print(f"✅ Статус анализа {analysis_id} обновлен на '{status}'")
        
        # Уведомляем о завершении анализа
        if status == "completed":
            print(f"🎉 Анализ {analysis_id} успешно завершен!")
        elif status == "failed":
            print(f"❌ Анализ {analysis_id} завершился с ошибкой")
            
        return analysis"""
        break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка в update_analysis_status исправлена!")
