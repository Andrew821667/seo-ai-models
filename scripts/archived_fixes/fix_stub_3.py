
import os

# Читаем файл  
with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

# Найдем заглушку в update_analysis_status (около строки 450)
for i, line in enumerate(lines):
    if i >= 448 and i <= 452 and "return None" in line and "analysis = self.get_analysis(analysis_id)" in lines[i-3]:
        # Заменяем заглушку
        lines[i] = """        
        # Сохраняем обновленный анализ
        self._save_analysis(analysis)
        
        print(f"✅ Статус анализа {analysis_id} обновлен на '{status}'")
        return analysis"""
        break

# Записываем обратно
with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #3 (update_analysis_status) исправлена!")
